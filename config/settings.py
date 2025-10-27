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

# Device configuration for models
# Options: "cuda", "cuda:0", "cuda:1", "cpu"
# When using vLLM, adjust --gpu-memory-utilization to leave space for embedding/reranker
# Recommended: vLLM with 0.70-0.75 utilization to leave ~25-30% for these models
def get_device_config():
    """
    Get device configuration with intelligent fallback.

    IMPORTANT: When vLLM is running, always use CPU for embeddings to avoid conflicts.
    The torch.cuda.memory_reserved() check is unreliable when vLLM is using the GPU.

    Returns:
        tuple: (embedding_device, reranker_device, device_info)
    """
    # Check if user explicitly wants CPU via environment variable
    force_cpu = os.environ.get('EMBEDDING_DEVICE', '').lower() == 'cpu'

    if force_cpu:
        return "cpu", "cpu", "CPU forced via EMBEDDING_DEVICE environment variable"

    if not torch.cuda.is_available():
        return "cpu", "cpu", "No CUDA available, using CPU"

    # Check available GPU memory
    try:
        # Get GPU memory info
        device_props = torch.cuda.get_device_properties(0)
        total_memory = device_props.total_memory / (1024**3)  # Convert to GB

        # Get currently allocated memory
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
        free_memory = total_memory - reserved_memory

        device_info = (f"GPU: {device_props.name}, Total: {total_memory:.2f}GB, "
                      f"Free: {free_memory:.2f}GB, Allocated: {allocated_memory:.2f}GB")

        # CRITICAL FIX: Be much more conservative with GPU usage
        # If ANY memory is already allocated/reserved, assume vLLM or another service is using the GPU
        # Default to CPU to avoid conflicts and crashes
        if reserved_memory > 1.0 or allocated_memory > 0.5:
            print(f"[INFO] GPU appears to be in use (reserved: {reserved_memory:.2f}GB, allocated: {allocated_memory:.2f}GB)")
            print(f"[INFO] Using CPU for embeddings/reranker to avoid conflicts with vLLM")
            return "cpu", "cpu", device_info

        # Only use GPU if there's substantial free memory AND nothing else is using it
        if free_memory >= 4.0:
            print(f"[INFO] Sufficient GPU memory available ({free_memory:.2f}GB free)")
            return "cuda", "cuda", device_info
        else:
            print(f"[INFO] Limited GPU memory ({free_memory:.2f}GB free, need ≥4GB for safe GPU usage)")
            print(f"[INFO] Using CPU for embeddings/reranker")
            return "cpu", "cpu", device_info

    except Exception as e:
        print(f"[WARNING] Error checking GPU memory: {e}")
        print(f"[INFO] Defaulting to CPU for safety")
        return "cpu", "cpu", f"Error checking GPU: {e}"

# Get device configuration at startup
EMBEDDING_DEVICE, RERANKER_DEVICE, DEVICE_INFO = get_device_config()
print(f"[DEVICE CONFIG] Embedding: {EMBEDDING_DEVICE}, Reranker: {RERANKER_DEVICE}")
print(f"[DEVICE INFO] {DEVICE_INFO}")

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