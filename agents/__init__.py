"""
Agents package for the Agentic RAG Medical Documentation System.
Provides retrieval agent implementation.
"""

from .retrieval_agent import (
    create_retrieval_agent,
    invoke_retrieval_agent,
)

__all__ = [
    'create_retrieval_agent',
    'invoke_retrieval_agent',
]