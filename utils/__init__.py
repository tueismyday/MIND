"""
Utilities package for the MIND medical documentation system.

This package provides essential utilities for text processing, PDF generation,
error handling, performance profiling, and validation reporting.

Modules:
    - text_processing: Parse and manipulate medical document text
    - text_patterns: Pre-compiled regex patterns for efficient text matching
    - pdf_utils: PDF generation with citations and formatting
    - pdf_styles: PDF styling configuration and constants
    - error_handling: Retry logic and safe wrappers for LLM/RAG operations
    - exceptions: Custom exception hierarchy
    - profiling: Performance monitoring decorators
    - token_tracker: Token usage tracking
    - validation_report_logger: Validation report generation

Example:
    >>> from utils import save_to_pdf, parse_section_subsections
    >>> from utils.exceptions import PDFGenerationError
    >>> from utils.profiling import profile
"""

# Exceptions (import first as other modules depend on them)
from .exceptions import (
    # Base exceptions
    UtilityError,

    # Text processing exceptions
    TextProcessingError,
    TextParsingError,
    DateParsingError,
    InvalidSectionFormatError,

    # PDF generation exceptions
    PDFGenerationError,
    PDFRenderingError,
    PDFStyleError,

    # Validation exceptions
    ValidationError,
    ValidationReportError,

    # Profiling exceptions
    ProfilingError,

    # Generation exceptions (backward compatibility)
    GenerationError,
    RAGError,
    LLMError,
    ParseError,
)

# Text patterns
from .text_patterns import (
    # Compiled patterns
    SECTION_HEADER_PATTERN,
    SUBSECTION_PATTERN,
    CITATION_PATTERN,
    WARNING_BOX_PATTERN,
    DATE_YMD_HMS_PATTERN,
    DATE_YMD_PATTERN,

    # Helper functions
    is_section_header,
    is_subsection_header,
    contains_citation,
    contains_warning,
    extract_section_title,
    extract_subsection_title,
    normalize_whitespace,
)

# Text processing
from .text_processing import (
    # Main functions
    parse_section_subsections,
    parse_medical_record_date,
    extract_final_section,
    assemble_document_from_sections,

    # Backward compatibility aliases
    split_section_into_subsections,
    parse_date_safe,
    assemble_final_document,

    # Data classes
    Subsection,
)

# PDF utilities
from .pdf_utils import (
    save_to_pdf,
    EnhancedPDFBuilder,
    CitationExtractor,
    detect_content_type,
    parse_enhanced_document,
)

# PDF styles
from .pdf_styles import (
    StyleFactory,
    PDFFonts,
    COLORS,
    LAYOUT,
    PDFColors,
    PDFLayout,
)

# Error handling
from .error_handling import (
    retry_on_failure,
    safe_rag_search,
    safe_llm_invoke,
)

# Profiling
from .profiling import (
    profile,
    profile_with_memory,
    log_execution_time,
)

# Token tracking
from .token_tracker import (
    TokenTracker,
    TokenUsage,
    get_token_tracker,
)

# Validation reporting
from .validation_report_logger import (
    ValidationReportLogger,
    ValidationMetrics,
    SubsectionReport,
    get_report_logger,
)

__all__ = [
    # Exceptions
    'UtilityError',
    'TextProcessingError',
    'TextParsingError',
    'DateParsingError',
    'InvalidSectionFormatError',
    'PDFGenerationError',
    'PDFRenderingError',
    'PDFStyleError',
    'ValidationError',
    'ValidationReportError',
    'ProfilingError',
    'GenerationError',
    'RAGError',
    'LLMError',
    'ParseError',

    # Text patterns
    'SECTION_HEADER_PATTERN',
    'SUBSECTION_PATTERN',
    'CITATION_PATTERN',
    'WARNING_BOX_PATTERN',
    'DATE_YMD_HMS_PATTERN',
    'DATE_YMD_PATTERN',
    'is_section_header',
    'is_subsection_header',
    'contains_citation',
    'contains_warning',
    'extract_section_title',
    'extract_subsection_title',
    'normalize_whitespace',

    # Text processing
    'parse_section_subsections',
    'parse_medical_record_date',
    'extract_final_section',
    'assemble_document_from_sections',
    'split_section_into_subsections',  # Backward compatibility
    'parse_date_safe',  # Backward compatibility
    'assemble_final_document',  # Backward compatibility
    'Subsection',

    # PDF utilities
    'save_to_pdf',
    'EnhancedPDFBuilder',
    'CitationExtractor',
    'detect_content_type',
    'parse_enhanced_document',

    # PDF styles
    'StyleFactory',
    'PDFFonts',
    'COLORS',
    'LAYOUT',
    'PDFColors',
    'PDFLayout',

    # Error handling
    'retry_on_failure',
    'safe_rag_search',
    'safe_llm_invoke',

    # Profiling
    'profile',
    'profile_with_memory',
    'log_execution_time',

    # Token tracking
    'TokenTracker',
    'TokenUsage',
    'get_token_tracker',

    # Validation reporting
    'ValidationReportLogger',
    'ValidationMetrics',
    'SubsectionReport',
    'get_report_logger',
]