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

# Device selection mode for each model
# Options: "auto" (automatic device selection with GPU memory check), "cpu" (force CPU), "cuda" (force GPU)
EMBEDDING_DEVICE_MODE = os.environ.get('EMBEDDING_DEVICE_MODE', 'auto').lower()
RERANKER_DEVICE_MODE = os.environ.get('RERANKER_DEVICE_MODE', 'auto').lower()

def get_device_config():
    """
    Get device configuration with intelligent fallback.

    Respects EMBEDDING_DEVICE_MODE and RERANKER_DEVICE_MODE settings:
    - "auto": Automatic device selection with GPU memory check
    - "cpu": Force CPU usage
    - "cuda": Force GPU usage (cuda:0)

    Uses torch.cuda.mem_get_info() which queries the CUDA driver directly
    to get ACTUAL free memory, accounting for vLLM or any other GPU usage.

    Returns:
        tuple: (embedding_device, reranker_device, device_info)
    """
    embedding_device = None
    reranker_device = None
    device_info_parts = []

    # Check legacy environment variable for backward compatibility
    force_cpu = os.environ.get('EMBEDDING_DEVICE', '').lower() == 'cpu'
    if force_cpu:
        print(f"[INFO] Legacy EMBEDDING_DEVICE=cpu detected, forcing CPU mode")
        return "cpu", "cpu", "CPU forced via EMBEDDING_DEVICE environment variable"

    # Handle embedding device mode
    if EMBEDDING_DEVICE_MODE == "cpu":
        embedding_device = "cpu"
        device_info_parts.append("Embedding: CPU (manual)")
        print(f"[DEVICE CONFIG] Embedding device: CPU (manual selection)")
    elif EMBEDDING_DEVICE_MODE == "cuda":
        if torch.cuda.is_available():
            embedding_device = "cuda"
            device_info_parts.append("Embedding: CUDA (manual)")
            print(f"[DEVICE CONFIG] Embedding device: CUDA (manual selection)")
        else:
            print(f"[WARNING] CUDA requested for embedding but not available, using CPU")
            embedding_device = "cpu"
            device_info_parts.append("Embedding: CPU (CUDA unavailable)")
    # else: auto mode, will be determined below

    # Handle reranker device mode
    if RERANKER_DEVICE_MODE == "cpu":
        reranker_device = "cpu"
        device_info_parts.append("Reranker: CPU (manual)")
        print(f"[DEVICE CONFIG] Reranker device: CPU (manual selection)")
    elif RERANKER_DEVICE_MODE == "cuda":
        if torch.cuda.is_available():
            reranker_device = "cuda"
            device_info_parts.append("Reranker: CUDA (manual)")
            print(f"[DEVICE CONFIG] Reranker device: CUDA (manual selection)")
        else:
            print(f"[WARNING] CUDA requested for reranker but not available, using CPU")
            reranker_device = "cpu"
            device_info_parts.append("Reranker: CPU (CUDA unavailable)")
    # else: auto mode, will be determined below

    # If both devices are already determined (manual mode), return early
    if embedding_device is not None and reranker_device is not None:
        device_info = ", ".join(device_info_parts)
        return embedding_device, reranker_device, device_info

    # Auto mode: Check CUDA availability and memory
    if not torch.cuda.is_available():
        final_embedding = embedding_device if embedding_device else "cpu"
        final_reranker = reranker_device if reranker_device else "cpu"
        device_info_parts.append("No CUDA available")
        return final_embedding, final_reranker, ", ".join(device_info_parts)

    # Check available GPU memory for auto mode devices
    try:
        # CRITICAL: Use mem_get_info() which queries CUDA driver directly
        # This shows ACTUAL free memory including vLLM usage
        free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(0)
        free_memory = free_memory_bytes / (1024**3)  # Convert to GB
        total_memory = total_memory_bytes / (1024**3)
        used_memory = total_memory - free_memory

        # Get device properties for name
        device_props = torch.cuda.get_device_properties(0)

        gpu_info = (f"GPU: {device_props.name}, Total: {total_memory:.2f}GB, "
                   f"Used: {used_memory:.2f}GB, Free: {free_memory:.2f}GB")

        print(f"[GPU MEMORY] {gpu_info}")

        # Memory requirements:
        # - Embedding model (Qwen3-Embedding-0.6B): ~1.2GB
        # - Reranker model (Qwen3-Reranker-0.6B): ~1.2GB
        # - Safety buffer: ~0.5GB
        # Total needed: ~3GB free

        MIN_FREE_MEMORY_GB = 2.5  # Minimum free memory to attempt GPU usage

        auto_device = None
        if free_memory >= MIN_FREE_MEMORY_GB:
            print(f"[GPU MEMORY] ✓ Sufficient free memory ({free_memory:.2f}GB ≥ {MIN_FREE_MEMORY_GB}GB)")
            print(f"[GPU MEMORY] ✓ Auto mode will use GPU")
            auto_device = "cuda"
        else:
            print(f"[GPU MEMORY] ✗ Insufficient free memory ({free_memory:.2f}GB < {MIN_FREE_MEMORY_GB}GB)")
            print(f"[GPU MEMORY] → Auto mode will use CPU")
            if used_memory > 5.0:
                print(f"[GPU MEMORY] → vLLM using {used_memory:.2f}GB")
                print(f"[GPU MEMORY] → Consider lowering vLLM --gpu-memory-utilization to 0.70-0.75")
            auto_device = "cpu"

        # Apply auto device to models in auto mode
        if embedding_device is None:
            embedding_device = auto_device
            device_info_parts.append(f"Embedding: {auto_device.upper()} (auto)")

        if reranker_device is None:
            reranker_device = auto_device
            device_info_parts.append(f"Reranker: {auto_device.upper()} (auto)")

        device_info_parts.append(gpu_info)
        return embedding_device, reranker_device, ", ".join(device_info_parts)

    except Exception as e:
        print(f"[WARNING] Error checking GPU memory: {e}")
        print(f"[INFO] Auto mode defaulting to CPU for safety")

        # Apply CPU to auto mode devices
        final_embedding = embedding_device if embedding_device else "cpu"
        final_reranker = reranker_device if reranker_device else "cpu"
        device_info_parts.append(f"Error checking GPU: {e}")

        return final_embedding, final_reranker, ", ".join(device_info_parts)

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

# vLLM configuration
# Mode: "server" (localhost API) or "local" (in-Python instance)
VLLM_MODE = os.environ.get('VLLM_MODE', 'server')  # "server" or "local"
VLLM_SERVER_URL = os.environ.get('VLLM_SERVER_URL', 'http://localhost:8000')
VLLM_MODEL_NAME = os.environ.get('VLLM_MODEL_NAME', 'cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit')

# vLLM local mode configuration (only used when VLLM_MODE="local")
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get('VLLM_GPU_MEMORY_UTILIZATION', '0.75'))
VLLM_MAX_MODEL_LEN = int(os.environ.get('VLLM_MAX_MODEL_LEN', '14000'))
VLLM_MAX_NUM_SEQS = int(os.environ.get('VLLM_MAX_NUM_SEQS', '1'))

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