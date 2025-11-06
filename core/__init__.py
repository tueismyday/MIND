"""
Core components package for the MIND medical documentation system.

This package provides fundamental infrastructure components including:
- Database management for vector databases
- Embedding and reranker model loading
- Memory management for conversation context
- Device management for GPU/CPU selection
- Custom exceptions for error handling
- Type definitions for improved type safety

Modules:
    database: Vector database management (ChromaDB)
    embeddings: HuggingFace embedding model loading
    reranker: Cross-encoder reranker model loading
    memory: Conversation memory and context management
    device_manager: GPU/CPU device selection and memory monitoring
    exceptions: Custom exception classes
    types: Type aliases and definitions

Quick Start:
    >>> from core import db_manager, get_embedding_model, memory_manager
    >>>
    >>> # Access databases
    >>> patient_db = db_manager.patient_db
    >>> guideline_db = db_manager.guideline_db
    >>>
    >>> # Get embedding model
    >>> embeddings = get_embedding_model()
    >>>
    >>> # Use memory manager
    >>> memory = memory_manager.retrieval_memory
"""

# Database management
from .database import (
    DatabaseManager,
    db_manager,  # Global singleton instance
)

# Embeddings (note: renamed from get_embeddings to get_embedding_model)
from .embeddings import get_embedding_model

# Reranker (note: renamed from get_reranker to get_reranker_model)
from .reranker import get_reranker_model

# Memory management
from .memory import (
    MemoryManager,
    memory_manager,  # Global singleton instance
)

# Device management
from .device_manager import (
    get_optimal_device,
    check_gpu_memory,
    cleanup_cuda_cache,
    log_device_info,
)

# Exceptions
from .exceptions import (
    CoreError,
    DatabaseConnectionError,
    ModelLoadingError,
    InsufficientGPUMemoryError,
    DeviceNotAvailableError,
)

# Type definitions
from .types import (
    DeviceType,
    MemoryInfo,
    DatabaseStats,
    ModelConfig,
)

# For backward compatibility, provide old function names
# These will be deprecated in future versions
get_embeddings = get_embedding_model
get_reranker = get_reranker_model

__all__ = [
    # Database
    'DatabaseManager',
    'db_manager',

    # Embeddings
    'get_embedding_model',
    'get_embeddings',  # Backward compatibility

    # Reranker
    'get_reranker_model',
    'get_reranker',  # Backward compatibility

    # Memory
    'MemoryManager',
    'memory_manager',

    # Device management
    'get_optimal_device',
    'check_gpu_memory',
    'cleanup_cuda_cache',
    'log_device_info',

    # Exceptions
    'CoreError',
    'DatabaseConnectionError',
    'ModelLoadingError',
    'InsufficientGPUMemoryError',
    'DeviceNotAvailableError',

    # Types
    'DeviceType',
    'MemoryInfo',
    'DatabaseStats',
    'ModelConfig',
]
