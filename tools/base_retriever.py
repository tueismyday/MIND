"""
Base retriever interface for the tools package.

This module defines the abstract base class and data structures for all
retrieval operations, providing a unified interface across different
retrieval strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class RetrievalResult:
    """
    Standard retrieval result format used across all retrieval tools.

    This class provides a consistent structure for search results regardless
    of the underlying retrieval strategy (semantic, keyword, hybrid, etc.).

    Attributes:
        content: The main text content of the retrieved document
        metadata: Additional information about the document
        score: Overall relevance score (0-100 scale)
        source_type: Origin of the result ("guideline", "patient", "document")
        timestamp: When the document was created/last modified
        entry_type: Type of entry (e.g., "Clinical Note", "Lab Result")
        document_id: Unique identifier for the document
        chunk_index: Position of this chunk within the document
        ranking_details: Strategy-specific ranking information
    """

    content: str
    metadata: Dict[str, Any]
    score: float
    source_type: str
    timestamp: Optional[str] = None
    entry_type: Optional[str] = None
    document_id: Optional[str] = None
    chunk_index: int = 0
    ranking_details: Dict[str, Any] = field(default_factory=dict)

    def get_snippet(self, max_length: int = 150) -> str:
        """
        Get a short snippet of the content.

        Args:
            max_length: Maximum length of the snippet

        Returns:
            Truncated content string
        """
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.

        Returns:
            Dictionary representation of the result
        """
        return {
            'content': self.content,
            'metadata': self.metadata,
            'score': self.score,
            'source_type': self.source_type,
            'timestamp': self.timestamp,
            'entry_type': self.entry_type,
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'ranking_details': self.ranking_details
        }


@dataclass
class SearchQuery:
    """
    Encapsulates a search query with its parameters.

    Attributes:
        text: The search query text
        top_k: Number of results to return
        filters: Optional metadata filters
        enable_reranking: Whether to apply reranking
        note_types: Specific note types to include (if applicable)
    """

    text: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    enable_reranking: bool = True
    note_types: Optional[List[str]] = None

    def __post_init__(self):
        """Validate query parameters."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if not self.text or not self.text.strip():
            raise ValueError("Query text cannot be empty")


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval implementations.

    This interface ensures consistency across different retrieval strategies
    including semantic search, keyword search, hybrid search, and RRF-based
    approaches.
    """

    @abstractmethod
    def retrieve(self, query: SearchQuery) -> List[RetrievalResult]:
        """
        Retrieve relevant documents based on the query.

        Args:
            query: SearchQuery object containing query text and parameters

        Returns:
            List of RetrievalResult objects sorted by relevance

        Raises:
            RetrievalError: If retrieval fails
            EmptyCorpusError: If no documents are indexed
            InvalidSearchParametersError: If query parameters are invalid
        """
        pass

    @abstractmethod
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for search.

        Args:
            documents: List of document dictionaries to index

        Raises:
            IndexingError: If indexing fails
        """
        pass

    def retrieve_with_sources(self, query: SearchQuery) -> tuple[str, List[RetrievalResult]]:
        """
        Retrieve documents and format with source references.

        This is a convenience method that calls retrieve() and formats the
        results with source citations.

        Args:
            query: SearchQuery object

        Returns:
            Tuple of (formatted_content, list_of_results)
        """
        results = self.retrieve(query)
        formatted_content = self._format_results(results, query)
        return formatted_content, results

    @abstractmethod
    def _format_results(self, results: List[RetrievalResult], query: SearchQuery) -> str:
        """
        Format retrieval results as human-readable text.

        Args:
            results: List of retrieval results
            query: Original search query

        Returns:
            Formatted string representation of results
        """
        pass

    def get_corpus_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed corpus.

        Returns:
            Dictionary with corpus statistics (size, last_updated, etc.)
        """
        return {
            "corpus_size": 0,
            "last_updated": None,
            "indexed": False
        }


class SemanticRetriever(BaseRetriever):
    """Base class for semantic (embedding-based) retrieval."""

    pass


class KeywordRetriever(BaseRetriever):
    """Base class for keyword (BM25-based) retrieval."""

    pass


class HybridRetriever(BaseRetriever):
    """Base class for hybrid retrieval (combining semantic + keyword)."""

    pass


@dataclass
class RRFSearchResult(RetrievalResult):
    """
    Extended result format for RRF (Reciprocal Rank Fusion) search.

    Includes additional RRF-specific ranking information beyond the
    standard RetrievalResult.

    Attributes:
        rrf_score: Normalized RRF fusion score (0-100)
        raw_rrf_score: Original RRF score before normalization
        semantic_score: Raw semantic similarity score
        keyword_score: Raw BM25 keyword score
        semantic_rank: Position in semantic ranking
        keyword_rank: Position in keyword ranking
        cross_encoder_score: Cross-encoder relevance score
        recency_boost: Temporal relevance boost
        rrf_k: RRF smoothing constant used
        rank_window: Rank window size used
    """

    rrf_score: float = 0.0
    raw_rrf_score: float = 0.0
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    semantic_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    cross_encoder_score: float = 0.0
    recency_boost: float = 0.0
    rrf_k: int = 60
    rank_window: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert RRF result to dictionary format.

        Returns:
            Dictionary representation including RRF-specific fields
        """
        base_dict = super().to_dict()
        base_dict.update({
            'rrf_score': self.rrf_score,
            'raw_rrf_score': self.raw_rrf_score,
            'semantic_score': self.semantic_score,
            'keyword_score': self.keyword_score,
            'semantic_rank': self.semantic_rank,
            'keyword_rank': self.keyword_rank,
            'cross_encoder_score': self.cross_encoder_score,
            'recency_boost': self.recency_boost,
            'rrf_k': self.rrf_k,
            'rank_window': self.rank_window
        })
        return base_dict
