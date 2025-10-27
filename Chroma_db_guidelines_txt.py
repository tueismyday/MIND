from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import sys
import shutil
import uuid
import re
from pathlib import Path

# Add the parent directory to the path to import settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE


def load_txt_from_directory(data_dir: str) -> list[dict]:
    """Load all TXT files from a directory.
    
    Args:
        data_dir: Directory path containing TXT files.
        
    Returns:
        List of dictionaries with 'content' and 'metadata' keys.
    """
    txt_documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[ERROR] Directory does not exist: {data_dir}")
        return []
    
    # Find all TXT files recursively, excluding checkpoint files
    txt_files = [
        f for f in data_path.rglob("*.txt") 
        if "-checkpoint" not in f.name
    ]
    
    if not txt_files:
        print(f"[WARNING] No TXT files found in: {data_dir}")
        return []
    
    print(f"[INFO] Found {len(txt_files)} TXT files to process")
    
    for txt_file in txt_files:
        print(f"[INFO] Processing: {txt_file.name}")
        
        file_handle = None
        try:
            # Read text file with UTF-8 encoding using context manager
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as file_handle:
                text_content = file_handle.read()
            # File is automatically closed here when exiting the 'with' block
            
            if text_content.strip():
                txt_documents.append({
                    'content': text_content,
                    'metadata': {
                        'source': str(txt_file),
                        'filename': txt_file.name,
                        'file_size': txt_file.stat().st_size
                    }
                })
                print(f"[SUCCESS] Loaded {len(text_content)} characters from {txt_file.name}")
            else:
                print(f"[WARNING] No content found in {txt_file.name}")
            
            # Explicitly delete the text content to free memory
            del text_content
                
        except Exception as e:
            print(f"[ERROR] Failed to process {txt_file.name}: {e}")
            continue
        finally:
            # Ensure file handle is closed even if an error occurs
            if file_handle is not None and not file_handle.closed:
                file_handle.close()
    
    print(f"[INFO] Successfully processed {len(txt_documents)} TXT files")
    return txt_documents


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


def load_and_process_txt(data_dir: str) -> list[dict]:
    """Load TXT files and extract structured sections based only on 'Overskrift:' headers.
    
    Args:
        data_dir: Directory path containing TXT files to process.
        
    Returns:
        A list of dictionaries, each containing extracted section text and metadata.
        Each dictionary has 'text' and 'metadata' keys.
    """
    # Load TXT files
    documents = load_txt_from_directory(data_dir)
    
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
        
        # Clean up memory after processing each document
        del full_text
        del structured_sections

    # Clean up the documents list to free memory
    del documents
    
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
    
    # Clean up temporary lists to free memory
    del texts
    del metadatas
    
    stored_chunks = vectordb.get()
    print(f"[DEBUG] Total stored chunks: {len(stored_chunks['documents'])}")
    
    return vectordb


def validate_directory_structure(data_dir: str) -> bool:
    """Validate that the data directory exists and contains TXT files.
    
    Args:
        data_dir: Path to the data directory.
        
    Returns:
        True if directory is valid and contains TXT files, False otherwise.
    """
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory does not exist: {data_dir}")
        return False
    
    txt_files = [
        f for f in Path(data_dir).rglob("*.txt") 
        if "-checkpoint" not in f.name
    ]
    
    if not txt_files:
        print(f"[ERROR] No TXT files found in directory: {data_dir}")
        return False
    
    print(f"[INFO] Found {len(txt_files)} TXT files in {data_dir}")
    return True


def main() -> None:
    """Main function to process TXT documents and create vector database."""
    data_dir = os.path.join(os.path.dirname(__file__), "data/hospital_guidelines") 
    db_dir = os.path.join(os.path.dirname(__file__), "data/hospital_guidelines_db")

    # Validate directory structure
    if not validate_directory_structure(data_dir):
        print("[ERROR] Cannot proceed without valid data directory")
        return

    print("Loading and processing TXT files...")
    chunks = load_and_process_txt(data_dir)
    
    if not chunks:
        print("[ERROR] No chunks were created from TXT processing")
        return
    
    print(f"Created {len(chunks)} chunks from TXT files")
    
    # Debugging: Display chunks before storing them in vector DB
    display_chunks(chunks)
    
    print("Creating vector store...")
    try:
        vectordb = create_vector_store(chunks, db_dir)
        print(f"Vector store created and persisted at {db_dir}")
        
        # Verify the vector store
        stored_data = vectordb.get()
        print(f"[SUCCESS] Vector store contains {len(stored_data['documents'])} documents")
        
        # Clean up
        del stored_data
        del vectordb
        
    except Exception as e:
        print(f"[ERROR] Failed to create vector store: {e}")
        return
    finally:
        # Final cleanup
        if 'chunks' in locals():
            del chunks
        print("[INFO] Memory cleanup completed")


if __name__ == "__main__":
    main()