"""
Vector database management for the Agentic RAG Medical Documentation System.
Handles ChromaDB initialization and operations.
"""

from langchain_chroma import Chroma
from .embeddings import get_embeddings
from config.settings import (
    GUIDELINE_DB_DIR, 
    PATIENT_DB_DIR, 
    GENERATED_DOCS_DB_DIR,
    ensure_directories
)

class DatabaseManager:
    """Manages all vector database instances."""
    
    def __init__(self):
        ensure_directories()
        self.embeddings = get_embeddings()
        self._patient_db = None
        self._guideline_db = None
        self._generated_docs_db = None
    
    @property
    def patient_db(self) -> Chroma:
        """Get the patient records vector database."""
        if self._patient_db is None:
            self._patient_db = Chroma(
                persist_directory=str(PATIENT_DB_DIR), 
                embedding_function=self.embeddings
            )
        return self._patient_db
    
    @property
    def guideline_db(self) -> Chroma:
        """Get the guidelines vector database."""
        if self._guideline_db is None:
            self._guideline_db = Chroma(
                persist_directory=str(GUIDELINE_DB_DIR), 
                embedding_function=self.embeddings
            )
        return self._guideline_db
    
    @property
    def generated_docs_db(self) -> Chroma:
        """Get the generated documents vector database."""
        if self._generated_docs_db is None:
            self._generated_docs_db = Chroma(
                persist_directory=str(GENERATED_DOCS_DB_DIR), 
                embedding_function=self.embeddings
            )
        return self._generated_docs_db
    
    def get_database_info(self) -> dict:
        """Get information about all databases."""
        try:
            patient_count = self.patient_db._collection.count()
        except:
            patient_count = 0
            
        try:
            guideline_count = self.guideline_db._collection.count()
        except:
            guideline_count = 0
            
        try:
            generated_count = self.generated_docs_db._collection.count()
        except:
            generated_count = 0
            
        return {
            "patient_record_chunks": patient_count,
            "guideline_chunks": guideline_count,
            "generated_document_chunks": generated_count
        }
    
    def print_database_info(self):
        """Print database information to console."""
        info = self.get_database_info()
        print(f"\n\n[INFO] The 'patient record' vector database currently holds {info['patient_record_chunks']} document chunks.")
        print(f"[INFO] The 'guideline' vector database currently holds {info['guideline_chunks']} document chunks")
        if info['generated_document_chunks'] > 0:
            print(f"[INFO] The 'generated documents' vector database currently holds {info['generated_document_chunks']} document chunks\n\n")
        else:
            print(f"[INFO] The 'generated documents' vector database currently holds 0 document chunks\n\n")

# Global database manager instance
db_manager = DatabaseManager()