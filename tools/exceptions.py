"""
Custom exceptions for the tools package.

This module defines domain-specific exception classes for retrieval operations,
providing clear error signaling and improved error handling throughout the tools.
"""


class RetrievalError(Exception):
    """Base exception for all retrieval-related errors."""

    def __init__(self, message: str, **context):
        """
        Initialize retrieval error with message and optional context.

        Args:
            message: Error message describing what went wrong
            **context: Additional context information (query, doc_id, etc.)
        """
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        """Return formatted error message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class EmptyCorpusError(RetrievalError):
    """Raised when no documents are available for search."""

    def __init__(self, corpus_name: str = "document corpus"):
        """
        Initialize empty corpus error.

        Args:
            corpus_name: Name of the corpus that is empty
        """
        super().__init__(
            f"No documents available in {corpus_name}",
            corpus_name=corpus_name
        )


class InvalidSearchParametersError(RetrievalError):
    """Raised when search parameters are invalid."""

    def __init__(self, parameter: str, value: any, reason: str):
        """
        Initialize invalid search parameters error.

        Args:
            parameter: Name of the invalid parameter
            value: Invalid value provided
            reason: Explanation of why the value is invalid
        """
        super().__init__(
            f"Invalid search parameter '{parameter}': {reason}",
            parameter=parameter,
            value=value,
            reason=reason
        )


class DateParsingError(RetrievalError):
    """Raised when document date parsing fails."""

    def __init__(self, date_str: str, doc_id: str = None):
        """
        Initialize date parsing error.

        Args:
            date_str: The date string that failed to parse
            doc_id: Optional document identifier
        """
        context = {"date_str": date_str}
        if doc_id:
            context["doc_id"] = doc_id

        super().__init__(
            f"Failed to parse date: '{date_str}'",
            **context
        )


class ModelLoadingError(RetrievalError):
    """Raised when a required model fails to load."""

    def __init__(self, model_name: str, reason: str = None):
        """
        Initialize model loading error.

        Args:
            model_name: Name/path of the model that failed to load
            reason: Optional explanation of the failure
        """
        message = f"Failed to load model: {model_name}"
        if reason:
            message += f" - {reason}"

        super().__init__(message, model_name=model_name, reason=reason)


class DocumentNotFoundError(RetrievalError):
    """Raised when a requested document cannot be found."""

    def __init__(self, doc_id: str, corpus: str = None):
        """
        Initialize document not found error.

        Args:
            doc_id: Document identifier that was not found
            corpus: Optional corpus name where document was searched
        """
        context = {"doc_id": doc_id}
        if corpus:
            context["corpus"] = corpus

        super().__init__(
            f"Document not found: {doc_id}",
            **context
        )


class RRFCalculationError(RetrievalError):
    """Raised when RRF calculation encounters an error."""

    def __init__(self, reason: str, **context):
        """
        Initialize RRF calculation error.

        Args:
            reason: Explanation of what went wrong during RRF calculation
            **context: Additional context (rankings, k value, etc.)
        """
        super().__init__(
            f"RRF calculation failed: {reason}",
            reason=reason,
            **context
        )


class CrossEncoderError(RetrievalError):
    """Raised when cross-encoder scoring fails."""

    def __init__(self, reason: str, num_pairs: int = None):
        """
        Initialize cross-encoder error.

        Args:
            reason: Explanation of the failure
            num_pairs: Optional number of query-document pairs being scored
        """
        context = {"reason": reason}
        if num_pairs is not None:
            context["num_pairs"] = num_pairs

        super().__init__(
            f"Cross-encoder scoring failed: {reason}",
            **context
        )


class IndexingError(RetrievalError):
    """Raised when document indexing fails."""

    def __init__(self, reason: str, num_documents: int = None):
        """
        Initialize indexing error.

        Args:
            reason: Explanation of the indexing failure
            num_documents: Optional number of documents being indexed
        """
        context = {"reason": reason}
        if num_documents is not None:
            context["num_documents"] = num_documents

        super().__init__(
            f"Document indexing failed: {reason}",
            **context
        )


class GuidelineRetrievalError(RetrievalError):
    """Raised when guideline retrieval fails."""

    def __init__(self, query: str, reason: str = None):
        """
        Initialize guideline retrieval error.

        Args:
            query: The query that failed
            reason: Optional explanation of the failure
        """
        message = f"Failed to retrieve guidelines for query: '{query}'"
        if reason:
            message += f" - {reason}"

        super().__init__(message, query=query, reason=reason)


class PatientDataRetrievalError(RetrievalError):
    """Raised when patient data retrieval fails."""

    def __init__(self, query: str, patient_id: str = None, reason: str = None):
        """
        Initialize patient data retrieval error.

        Args:
            query: The query that failed
            patient_id: Optional patient identifier
            reason: Optional explanation of the failure
        """
        context = {"query": query}
        if patient_id:
            context["patient_id"] = patient_id
        if reason:
            context["reason"] = reason

        message = f"Failed to retrieve patient data for query: '{query}'"
        if reason:
            message += f" - {reason}"

        super().__init__(message, **context)
