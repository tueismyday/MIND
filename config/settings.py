"""
Configuration settings for the Agentic RAG Medical Documentation System.
Contains all constants, paths, and configuration parameters.
"""

import os
from pathlib import Path
import torch

# Base project directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
GUIDELINE_DIR = DATA_DIR / "hospital_guidelines"
PATIENT_RECORD_DIR = DATA_DIR / "patient_record"
GENERATED_DOCS_DIR = DATA_DIR / "generated_documents"

# Vector database directories
GUIDELINE_DB_DIR = DATA_DIR / "hospital_guidelines_db"
PATIENT_DB_DIR = DATA_DIR / "patient_record_db"
GENERATED_DOCS_DB_DIR = DATA_DIR / "generated_documents_db"

# Cache directories
CACHE_DIR = DATA_DIR / "sentence_transformers_cache"

# Embedding model configuration
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
EMBEDDING_DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else 
RERANKER_DEVICE = "cpu"

# Agent configuration
MAX_AGENT_ITERATIONS = 10
AGENT_EARLY_STOPPING_METHOD = "generate"
MEMORY_MAX_TOKEN_LIMIT = 12000
CHAT_HISTORY_LIMIT = 20

# Retrieval configuration
INITIAL_RETRIEVAL_K = 20
FINAL_RETRIEVAL_K = 10
SIMILARITY_SCORE_THRESHOLD = 0.5

# Tool-specific search k values
GUIDELINE_SEARCH_K = 5  # k value for guideline searches in tools
GENERATED_DOC_SEARCH_K = 5  # k value for generated document searches in tools
PATIENT_SEARCH_K = 5  # k value for patient record searches in tools

# Generation configuration
GENERATION_TEMPERATURE = 0.2
CRITIQUE_TEMPERATURE = 0.1

# Validation configuration
DEFAULT_VALIDATION_CYCLES = 2  # Default number of validation/revision cycles per subsection
MAX_VALIDATION_CYCLES = 3      # Maximum allowed validation cycles
MIN_VALIDATION_CYCLES = 1      # Minimum validation cycles (at least one critique cycle)

# Hybrid approach configuration
USE_HYBRID_MULTI_FACT_APPROACH = True  # Enable multi-fact retrieval
FACT_COMPLEXITY_THRESHOLD = 4  # Subsections with >4 facts use multi-fact retrieval
MAX_SOURCES_PER_FACT = 2  # Max sources to retrieve per individual fact

# PDF configuration
PDF_FONT_PATH = "DejaVuSans.ttf"
DEFAULT_OUTPUT_NAME = "generated_medical_document.pdf"

# Default patient file (if available)
DEFAULT_PATIENT_FILE = PATIENT_RECORD_DIR / "Patient_journal_Geriatrisk_patient.pdf"

# Date parsing formats for patient records
DATE_FORMATS = ["%y.%m.%d %H:%M", "%y.%m.%d"]

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR, GUIDELINE_DIR, PATIENT_RECORD_DIR, GENERATED_DOCS_DIR,
        GUIDELINE_DB_DIR, PATIENT_DB_DIR, GENERATED_DOCS_DB_DIR, CACHE_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
def get_patient_file_path():
    """Get the path to the patient file if it exists."""
    if DEFAULT_PATIENT_FILE.exists():
        return str(DEFAULT_PATIENT_FILE)
    return None