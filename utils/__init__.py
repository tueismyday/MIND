"""
Utilities package for the Agentic RAG Medical Documentation System.
Provides text processing, PDF handling, profiling, and error handling.
"""

# Text processing
from .text_processing import (
    split_section_into_subsections,
    parse_date_safe,
    extract_final_section,
    assemble_final_document,
)

# PDF utilities
from .pdf_utils import (
    save_to_pdf,
    EnhancedPDFBuilder,
    CitationExtractor,
)

# Error handling
from .error_handling import (
    # Exceptions
    GenerationError,
    RAGError,
    LLMError,
    ParseError,
    
    # Utilities
    retry_on_failure,
    safe_rag_search,
    safe_llm_invoke,
)

# Profiling (if exists)
try:
    from .profiling import profile
    _has_profiling = True
except ImportError:
    # Profiling module might not exist
    def profile(func):
        """Dummy profiler decorator"""
        return func
    _has_profiling = False

__all__ = [
    # Text processing
    'split_section_into_subsections',
    'parse_date_safe',
    'extract_final_section',
    'assemble_final_document',
    
    # PDF utilities
    'extract_text_from_pdf',
    'save_to_pdf',
    'save_enhanced_pdf',
    'EnhancedPDFBuilder',
    'CitationExtractor',
    
    # Error handling
    'GenerationError',
    'RAGError',
    'LLMError',
    'ParseError',
    'retry_on_failure',
    'safe_rag_search',
    'safe_llm_invoke',
    
    # Profiling
    'profile',
]