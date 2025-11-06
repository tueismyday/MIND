"""
Custom exception classes for the MIND utility modules.

This module defines a hierarchy of exceptions for different error scenarios
in text processing, PDF generation, validation reporting, and profiling.

Exception Hierarchy:
    UtilityError (base)
    ├── TextProcessingError
    │   ├── TextParsingError
    │   ├── DateParsingError
    │   └── InvalidSectionFormatError
    ├── PDFGenerationError
    │   ├── PDFRenderingError
    │   └── PDFStyleError
    ├── ValidationError
    │   └── ValidationReportError
    └── ProfilingError

Example:
    >>> from utils.exceptions import DateParsingError
    >>> raise DateParsingError("Invalid date format: 'abc'")
"""

from typing import Optional


class UtilityError(Exception):
    """
    Base exception for all utility module errors.

    All custom exceptions in the utils module inherit from this class.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        """
        Initialize utility error.

        Args:
            message: Human-readable error message
            details: Optional additional context or diagnostic information
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """Return formatted error message with details if available."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# Text Processing Exceptions

class TextProcessingError(UtilityError):
    """Base exception for text processing errors."""
    pass


class TextParsingError(TextProcessingError):
    """
    Raised when text parsing operations fail.

    This includes failures in section splitting, content extraction,
    or any text manipulation that produces invalid results.

    Example:
        >>> raise TextParsingError(
        ...     "Failed to parse section markers",
        ...     details="No 'Overskrift:' markers found in text"
        ... )
    """
    pass


class DateParsingError(TextProcessingError):
    """
    Raised when date string parsing fails.

    This exception is raised when all known date format patterns
    fail to parse a date string from medical records.

    Example:
        >>> raise DateParsingError(
        ...     "Unable to parse date: 'invalid_date'",
        ...     details="Tried formats: ['%y.%m.%d %H:%M', '%y.%m.%d']"
        ... )
    """
    pass


class InvalidSectionFormatError(TextParsingError):
    """
    Raised when document section format is invalid or malformed.

    This exception indicates that the document structure doesn't match
    expected section markers and formatting conventions.

    Example:
        >>> raise InvalidSectionFormatError(
        ...     "Subsection marker format is invalid",
        ...     details="Expected 'Sub_section: Title', got 'Sub_section'"
        ... )
    """
    pass


# PDF Generation Exceptions

class PDFGenerationError(UtilityError):
    """
    Base exception for PDF generation errors.

    Raised when PDF document creation, rendering, or file operations fail.
    """
    pass


class PDFRenderingError(PDFGenerationError):
    """
    Raised when PDF content rendering fails.

    This includes failures in converting text to PDF elements,
    applying styles, or building the final PDF document.

    Example:
        >>> raise PDFRenderingError(
        ...     "Failed to render paragraph",
        ...     details="Invalid ReportLab style object"
        ... )
    """
    pass


class PDFStyleError(PDFGenerationError):
    """
    Raised when PDF styling operations fail.

    This includes failures in font loading, style creation,
    or color/layout configuration.

    Example:
        >>> raise PDFStyleError(
        ...     "Font loading failed",
        ...     details="DejaVu font file not found at path"
        ... )
    """
    pass


# Validation Exceptions

class ValidationError(UtilityError):
    """
    Base exception for validation-related errors.

    This exception and its subclasses cover errors in validation
    report generation and logging operations.
    """
    pass


class ValidationReportError(ValidationError):
    """
    Raised when validation report generation fails.

    This includes failures in creating PDF validation reports,
    collecting metrics, or formatting validation data.

    Example:
        >>> raise ValidationReportError(
        ...     "Failed to generate validation PDF",
        ...     details="No validation cycles recorded"
        ... )
    """
    pass


# Profiling Exceptions

class ProfilingError(UtilityError):
    """
    Raised when performance profiling operations fail.

    This includes failures in timing measurements, memory profiling,
    or report generation.

    Example:
        >>> raise ProfilingError(
        ...     "Memory profiling failed",
        ...     details="tracemalloc not available"
        ... )
    """
    pass


# Generation Exceptions (moved from error_handling.py for consistency)

class GenerationError(Exception):
    """
    Base exception for document generation errors.

    Used in error_handling.py for backward compatibility.
    """
    pass


class RAGError(GenerationError):
    """
    Raised when RAG (Retrieval-Augmented Generation) search fails.

    This indicates failures in retrieving relevant information
    from vector databases or search operations.
    """
    pass


class LLMError(GenerationError):
    """
    Raised when LLM invocation fails.

    This indicates failures in language model API calls,
    response processing, or token generation.
    """
    pass


class ParseError(GenerationError):
    """
    Raised when parsing or extraction operations fail.

    This indicates failures in extracting structured information
    from LLM responses or other text sources.
    """
    pass
