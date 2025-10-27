from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import sys
import shutil
import uuid
import re
import PyPDF2
from pathlib import Path

# Add the parent directory to the path to import settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Extracted text from all pages of the PDF.
        
    Raises:
        Exception: If PDF cannot be processed.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                print(f"[WARNING] PDF is encrypted, skipping: {pdf_path}")
                return ""
            
            # Check if PDF has pages
            if len(reader.pages) == 0:
                print(f"[WARNING] PDF has no pages, skipping: {pdf_path}")
                return ""
            
            # Extract text from all pages
            text_pages = []
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        # Clean up extracted text
                        page_text = page_text.replace('\x00', '').replace('\x0c', '\n')
                        page_text = page_text.encode('utf-8', errors='ignore').decode('utf-8')
                        text_pages.append(page_text)
                except Exception as e:
                    print(f"[WARNING] Failed to extract text from page {page_num + 1} in {pdf_path}: {e}")
                    continue
            
            if not text_pages:
                print(f"[WARNING] No text could be extracted from: {pdf_path}")
                return ""
                
            return "\n\n".join(text_pages)
            
    except Exception as e:
        print(f"[ERROR] Failed to process PDF {pdf_path}: {e}")
        return ""


def load_pdfs_from_directory(data_dir: str) -> list[dict]:
    """Load all PDF files from a directory using PyPDF2.
    
    Args:
        data_dir: Directory path containing PDF files.
        
    Returns:
        List of dictionaries with 'content' and 'metadata' keys.
    """
    pdf_documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[ERROR] Directory does not exist: {data_dir}")
        return []
    
    # Find all PDF files recursively
    pdf_files = [
    f for f in data_path.rglob("*.pdf") 
    if "-checkpoint" not in f.name
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


def split_by_section(text: str) -> list[tuple[str, str]]:
    """Split text into sections based on 'Overskrift:' lines.
    
    Args:
        text: The document text to split.
        
    Returns:
        A list of tuples, each containing (section_title, section_content).
        If no 'Overskrift:' lines are found, returns a single tuple with 
        ("Full Document", full_text).
    """
    sections = []
    last_known_section = None  # Track last detected section title
    current_text = []
    found_section_headers = False

    lines = text.split("\n")
    for line in lines:
        line = line.strip()

        # Detect section headers using "Overskrift:"
        match = re.match(r"Overskrift:\s*'?(.*?)'?$", line)

        if match:
            found_section_headers = True  # At least one header was found

            # Store previous section if we already have content
            if current_text:
                sections.append((last_known_section, "\n".join(current_text)))

            last_known_section = match.group(1).strip()
            current_text = []  # Start a new section
        else:
            if last_known_section is None:
                last_known_section = "Unknown Section"
            current_text.append(line)

    if current_text:
        sections.append((last_known_section, "\n".join(current_text)))

    if not found_section_headers:
        print("[INFO] No 'Overskrift:' sections found, treating full document as one chunk.")
        return [("Full Document", text.strip())]

    return sections

def load_and_process_pdfs(data_dir: str) -> list[dict]:
    """Load PDFs and extract structured sections based only on 'Overskrift:' headers.
    
    Args:
        data_dir: Directory path containing PDF files to process.
        
    Returns:
        A list of dictionaries, each containing extracted section text and metadata.
        Each dictionary has 'text' and 'metadata' keys.
    """
    # Load PDFs using PyPDF2
    documents = load_pdfs_from_directory(data_dir)
    
    if not documents:
        print("[ERROR] No documents could be loaded")
        return []

    all_chunks = []

    print(f"\nStarting vectorization process: \n - Creating chunks based only on 'Overskrift:' sections\n")
    
    for doc in documents:
        doc_id = str(uuid.uuid4())  # Unique ID for the document
        doc_name = doc['metadata']['filename']
        full_text = doc['content']
        
        structured_sections = split_by_section(full_text)
        
        print(f"[DEBUG] Found {len(structured_sections)} sections in document {doc_name} (ID: {doc_id})")
        
        is_full_doc = len(structured_sections) == 1 and structured_sections[0][0] == "Full Document"
        print(f"[DEBUG] {'No sections found' if is_full_doc else f'Found {len(structured_sections)} sections'} in document {doc_name} (ID: {doc_id})")

        # Store each section as a separate chunk without further splitting
        for i, (section_title, section_text) in enumerate(structured_sections):
            all_chunks.append({
                "text": section_text,
                "metadata": {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "section_title": section_title,
                    "section_index": i,
                    "granularity": "section",
                    "source": doc['metadata']['source'],
                    "file_size": doc['metadata']['file_size']
                }
            })

    return all_chunks

def display_chunks(chunks: list[dict]) -> None:
    """Display extracted chunks for debugging purposes.
    
    Args:
        chunks: List of chunk dictionaries containing text and metadata.
    """
    print("\n[DEBUG] Displaying Extracted Chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}/{len(chunks)}:")
        print(f"Document: {chunk['metadata']['document_name']}")
        print(f"Section: {chunk['metadata']['section_title']}")
        print(f"Section Index: {chunk['metadata']['section_index']}")
        print(f"Source: {chunk['metadata']['source']}")
        print(f"File Size: {chunk['metadata']['file_size']} bytes")
        print(f"Text Preview: {chunk['text'][:100]}...")  # Print first 100 characters for readability
        print("-" * 80)

def create_vector_store(chunks: list[dict], persist_directory: str) -> Chroma:
    """Create and persist ChromaDB vector store with section metadata.
    
    Args:
        chunks: List of chunk dictionaries containing text and metadata.
        persist_directory: Directory path to store the vector database.
        
    Returns:
        A Chroma vector store instance containing the embedded chunks.
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
    """Main function to process PDF documents and create vector database."""
    data_dir = os.path.join(os.path.dirname(__file__), "data/hospital_guidelines") 
    db_dir = os.path.join(os.path.dirname(__file__), "data/hospital_guidelines_db")

    # Validate directory structure
    if not validate_directory_structure(data_dir):
        print("[ERROR] Cannot proceed without valid data directory")
        return

    print("Loading and processing PDFs with PyPDF2...")
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