"""
Core components package for the Agentic RAG Medical Documentation System.
Provides database management, embeddings, and memory management.
"""

# Database management
from .database import (
    DatabaseManager,
    db_manager,  # Global instance
)

# Embeddings
from .embeddings import get_embeddings

# Memory management
from .memory import (
    MemoryManager,
    memory_manager,  # Global instance
)

__all__ = [
    # Database
    'DatabaseManager',
    'db_manager',
    
    # Embeddings
    'get_embeddings',
    
    # Memory
    'MemoryManager',
    'memory_manager',
]