"""
Custom exceptions for the document generation pipeline.

This module defines domain-specific exception classes for better error
handling and more informative error messages throughout the generation process.
"""


class GenerationError(Exception):
    """Base exception for all generation-related errors.

    This is the parent class for all custom exceptions in the generation pipeline.
    Catching this exception will catch all generation-specific errors.

    Attributes:
        message: Human-readable error message
        context: Optional dictionary with additional context about the error
    """

    def __init__(self, message: str, context: dict = None):
        """Initialize the GenerationError.

        Args:
            message: Human-readable error message
            context: Optional dictionary with additional context (e.g., section_title, patient_id)
        """
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return string representation with context if available."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class FactParsingError(GenerationError):
    """Raised when fact parsing from guidelines fails.

    This exception is raised when the LLM fails to parse guidelines into
    structured facts, or when the JSON response cannot be parsed.

    Example:
        >>> raise FactParsingError(
        ...     "Failed to parse JSON from LLM response",
        ...     context={"subsection": "Medication", "attempt": 3}
        ... )
    """
    pass


class FactRetrievalError(GenerationError):
    """Raised when RAG retrieval fails for a fact.

    This exception is raised when the RAG system fails to retrieve sources
    for a specific fact, either due to technical issues or insufficient data.

    Example:
        >>> raise FactRetrievalError(
        ...     "No sources found for fact query",
        ...     context={"query": "diabetes medication", "note_types": ["Medical"]}
        ... )
    """
    pass


class ValidationError(GenerationError):
    """Raised when validation fails critically.

    This exception is raised when the validation process encounters a critical
    error that prevents it from continuing, not when validation simply finds issues.

    Example:
        >>> raise ValidationError(
        ...     "Validation LLM failed after 3 retries",
        ...     context={"subsection": "Treatment Plan", "cycle": 2}
        ... )
    """
    pass


class InsufficientDataError(GenerationError):
    """Raised when patient data is insufficient for generation.

    This exception is raised when there is not enough patient data to generate
    meaningful content for a section or subsection.

    Example:
        >>> raise InsufficientDataError(
        ...     "No patient records found for required note types",
        ...     context={"note_types": ["Nursing", "Medical"], "patient_id": "12345"}
        ... )
    """
    pass


class AssemblyError(GenerationError):
    """Raised when subsection assembly from facts fails.

    This exception is raised when the LLM fails to assemble individual fact
    answers into a coherent subsection.

    Example:
        >>> raise AssemblyError(
        ...     "Failed to assemble facts into subsection",
        ...     context={"subsection": "Current Medications", "facts_count": 5}
        ... )
    """
    pass


class DocumentGenerationError(GenerationError):
    """Raised when overall document generation fails.

    This exception is raised for high-level document generation failures
    that affect the entire document rather than individual sections.

    Example:
        >>> raise DocumentGenerationError(
        ...     "Failed to generate document after section 3",
        ...     context={"sections_completed": 3, "total_sections": 8}
        ... )
    """
    pass


class ConfigurationError(GenerationError):
    """Raised when configuration is invalid or missing.

    This exception is raised when required configuration parameters are
    missing or invalid.

    Example:
        >>> raise ConfigurationError(
        ...     "Invalid max_validation_cycles value",
        ...     context={"value": -1, "min": 1, "max": 5}
        ... )
    """
    pass


class LLMInvocationError(GenerationError):
    """Raised when LLM invocation fails after all retries.

    This exception is raised when an LLM call fails after exhausting all
    retry attempts, indicating a persistent technical issue.

    Example:
        >>> raise LLMInvocationError(
        ...     "LLM failed after 3 retries with timeout",
        ...     context={"operation": "fact_answering", "model": "llama-3.1-70b"}
        ... )
    """
    pass


class SourceFormattingError(GenerationError):
    """Raised when source formatting or processing fails.

    This exception is raised when there are issues formatting sources for
    prompts or creating source references.

    Example:
        >>> raise SourceFormattingError(
        ...     "Missing required fields in source document",
        ...     context={"missing_fields": ["timestamp", "entry_type"]}
        ... )
    """
    pass
