from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import sys
import shutil
import uuid
import re
from pathlib import Path
from datetime import datetime
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

# Add the parent directory to the path to import settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, DANISH_CLINICAL_CATEGORIES, DATE_FORMATS


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using Docling and remove identification lines.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Extracted text with identification lines removed.
        
    Raises:
        Exception: If PDF cannot be processed.
    """
    try:
        num_threads = 4
        
        accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=AcceleratorDevice.AUTO
        )
        
        pipeline_options = PdfPipelineOptions(
            accelerator_options=accelerator_options, 
            do_ocr=True
        )
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        result = converter.convert(pdf_path)
        text = result.document.export_to_text()
        
        # Remove ID lines: '## 123456-7890 - NAME - AGE'
        text = re.sub(r'^##\s*\d{6}-\d{4}\s*-\s*[^-\n]+?\s*-\s*[^-\n]+\s*\n?', '', text, flags=re.MULTILINE)
        
        return text
        
    except Exception as e:
        print(f"[ERROR] Failed to process PDF {pdf_path}: {e}")
        return ""


def load_pdfs_from_directory(data_dir: str) -> list[dict]:
    """Load all PDF files from a directory using Docling."""
    
    pdf_documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[ERROR] Directory does not exist: {data_dir}")
        return []
    
    # Find all PDF files, EXCLUDING checkpoint files
    pdf_files = [
        f for f in data_path.rglob("*.pdf") 
        if "-checkpoint" not in f.name  # â† ADD THIS
    ]
    
    if not pdf_files:
        print(f"[WARNING] No PDF files found in: {data_dir}")
        return []
    
    print(f"[INFO] Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"[INFO] Processing: {pdf_file.name}")
        
        try:
            text_content = extract_text_from_pdf(str(pdf_file))
            
            if text_content.strip():
                pdf_documents.append({
                    'content': text_content,
                    'metadata': {
                        'source': str(pdf_file),
                        'filename': pdf_file.name,
                        'file_size': pdf_file.stat().st_size
                    }
                })
                print(f"[SUCCESS] Extracted {len(text_content)} characters from {pdf_file.name}")
            else:
                print(f"[WARNING] No content extracted from {pdf_file.name}")
                
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_file.name}: {e}")
            continue
    
    print(f"[INFO] Successfully processed {len(pdf_documents)} PDF files")
    return pdf_documents


def extract_date_from_text(text: str) -> str:
    """
    Extract the most recent date from text in DD.MM.YY format.
    
    Args:
        text: The text to search for dates
        
    Returns:
        str: The most recent date found in YYYY-MM-DD HH:MM format, or empty string if none found
    """
    matches = re.findall(r"\d{2}\.\d{2}\.\d{2}(?: \d{2}:\d{2})?", text)
    
    if not matches:
        return ""

    parsed_dates = []
    for m in matches:
        for fmt in ("%d.%m.%y %H:%M", "%d.%m.%y"):  # <--- DAY.MONTH.YEAR enforced
            try:
                dt = datetime.strptime(m, fmt)
                if dt.year > 2025:
                    continue
                parsed_dates.append(dt)
                break
            except ValueError:
                continue

    if not parsed_dates:
        return ""

    most_recent = max(parsed_dates)
    return most_recent.strftime("%Y-%m-%d %H:%M")

def infer_category(text: str) -> str:
    """
    Infer patient record category from Danish guideline keywords.
    
    Args:
        text: The text content to categorize
        
    Returns:
        str: The inferred category name, or "ukategoriseret" if no match found
    """
    text_lower = text.lower()
    for category, keywords in DANISH_CLINICAL_CATEGORIES.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "ukategoriseret"

def split_by_date(text: str) -> list[tuple[str, str]]:
    """
    Splits text by note timestamps in the format:
    dd.mm.yy tt:tt, <name>, <profession>, <yard>
    """
    
    # Pattern: DD.MM.YY HH:MM, followed by comma-separated fields
    pattern = r"(\d{2}\.\d{2}\.\d{2} \d{2}:\d{2}),\s*([^,\n]+),\s*([^,\n]+),\s*([^\n]+)"
    
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return [("Unknown", text.strip())]
    
    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        date_str = matches[i].group(1)  # DD.MM.YY HH:MM
        
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        
        chunks.append((date_str, chunk_text))
    
    return chunks

def extract_entry_type(text: str) -> str:
    """
    Extract the profession from the first line.

    Example:
        From: "01.01.11 11:11, Name, Læge, Afdeling A\n..."
        Extracts: "Læge"

    Args:
        text: The text chunk containing timestamp and header line

    Returns:
        str: The extracted profession, or 'Note' if not found
    """
    pattern = r"(\d{2}\.\d{2}\.\d{2} \d{2}:\d{2}),\s*([^,\n]+),\s*([^,\n]+),\s*([^\n]+)"
    lines = text.strip().splitlines()
    if not lines:
        return "Ukendt"

    match = re.match(pattern, lines[0])
    if match:
        return match.group(3).strip()  # Profession is the 3rd group

    return "Ukendt"

def extract_yard(text: str) -> str:
    """
    Extract the yard/department from the first line.

    Example:
        From: "01.01.11 11:11, Name, Læge, Afdeling A\n..."
        Extracts: "Afdeling A"

    Args:
        text: The text chunk containing timestamp and header line

    Returns:
        str: The extracted yard, or empty string if not found
    """
    pattern = r"(\d{2}\.\d{2}\.\d{2} \d{2}:\d{2}),\s*([^,\n]+),\s*([^,\n]+),\s*([^\n]+)"
    lines = text.strip().splitlines()
    if not lines:
        return ""

    match = re.match(pattern, lines[0])
    if match:
        return match.group(4).strip()  # Yard is the 4th group

    return ""

def load_and_process_pdfs(data_dir: str):
    """Load PDFs and split by date/timestamp using Docling"""
    
    # Load PDFs using Docling
    documents = load_pdfs_from_directory(data_dir)
    
    if not documents:
        print("[ERROR] No documents could be loaded")
        return []

    # Group documents by filename
    doc_texts = {}
    for doc in documents:
        doc_name = doc['metadata']['filename']
        if doc_name not in doc_texts:
            doc_texts[doc_name] = []
        doc_texts[doc_name].append(doc['content'].strip())

    all_chunks = []

    print(f"\nStarting vectorization process: \n - Chunking by timestamps \n - Assigning category and metadata \n - Generating summaries\n")
    
    for doc_name, pages in doc_texts.items():
        doc_id = str(uuid.uuid4())
        full_text = "\n".join(pages)
        
        dated_chunks = split_by_date(full_text)
        print(f"[DEBUG] Found {len(dated_chunks)} timestamp chunks in document {doc_name} (ID: {doc_id})")
        
        for i, (date_str, chunk_text) in enumerate(dated_chunks):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "chunk_index": i,
                    "granularity": "timestamp_chunk",
                    "category": infer_category(chunk_text),
                    "date": extract_date_from_text(date_str),
                    "entry_type": extract_entry_type(chunk_text),
                    "yard": extract_yard(chunk_text)
                }
            })

    return all_chunks

def display_chunks(chunks: list[dict]) -> None:
    """
    Display extracted chunks for debugging purposes.
    
    Args:
        chunks: List of chunk dictionaries containing text and metadata
    """
    
    print("\n[DEBUG] Displaying Extracted Chunks:\n")
    for chunk in chunks:
        print(f"Document: {chunk['metadata']['document_name']}")
        print(f"Chunk Index: {chunk['metadata']['chunk_index']}")
        print(f"Date: {chunk['metadata']['date']}")
        print(f"Category: {chunk['metadata']['category']}")
        print(f"Entry Type: {chunk['metadata']['entry_type']}")
        print(f"Yard: {chunk['metadata']['yard']}")
        print(f"Text Preview: {chunk['text'][:200]}...")  # Display first 200 characters
        print("-" * 80)

def create_vector_store(chunks: list[dict], persist_directory: str) -> Chroma:
    """
    Create and persist ChromaDB vector store with enhanced metadata.
    
    Args:
        chunks: List of chunk dictionaries containing text and metadata
        persist_directory: Directory path where the vector store will be saved
        
    Returns:
        Chroma: The created Chroma vector database instance
    """
    
    if os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory}")
        shutil.rmtree(persist_directory)
    
    # Use embedding model from settings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': EMBEDDING_DEVICE}
    )
    
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    print(f"Creating new vector store using {EMBEDDING_MODEL_NAME} on {EMBEDDING_DEVICE}...")
    vectordb = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    stored_chunks = vectordb.get()
    print(f"[DEBUG] Total stored chunks: {len(stored_chunks['documents'])}")
    return vectordb

def validate_directory_structure(data_dir: str) -> bool:
    """Validate that the data directory exists and contains PDF files.
    
    Args:
        data_dir: Path to the data directory.
        
    Returns:
        True if directory is valid and contains PDFs, False otherwise.
    """
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory does not exist: {data_dir}")
        return False
    
    pdf_files = list(Path(data_dir).rglob("*.pdf"))
    if not pdf_files:
        print(f"[ERROR] No PDF files found in directory: {data_dir}")
        return False
    
    print(f"[INFO] Found {len(pdf_files)} PDF files in {data_dir}")
    return True

def main() -> None:
    """
    Main function to execute the PDF processing and vector store creation workflow.
    """
    
    data_dir = os.path.join(os.path.dirname(__file__), "data/patient_record")
    db_dir = os.path.join(os.path.dirname(__file__), "data/patient_record_db")

    # Validate directory structure
    if not validate_directory_structure(data_dir):
        print("[ERROR] Cannot proceed without valid data directory")
        return

    print("Loading and processing PDFs with Docling...")
    chunks = load_and_process_pdfs(data_dir)
    
    if not chunks:
        print("[ERROR] No chunks were created from PDF processing")
        return
    
    print(f"Created {len(chunks)} chunks from PDFs")
    
    # Debugging: Display chunks before storing them in vector DB
    display_chunks(chunks)
    
    print("Creating vector store...")
    try:
        vectordb = create_vector_store(chunks, db_dir)
        print(f"Vector store created and persisted at {db_dir}")
        
        # Verify the vector store
        stored_data = vectordb.get()
        print(f"[SUCCESS] Vector store contains {len(stored_data['documents'])} documents")
        
    except Exception as e:
        print(f"[ERROR] Failed to create vector store: {e}")
        return

if __name__ == "__main__":
    main()