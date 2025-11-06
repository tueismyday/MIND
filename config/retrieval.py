"""
Retrieval configuration for the MIND medical documentation system.

This module configures RAG (Retrieval-Augmented Generation) parameters,
including search parameters, similarity thresholds, and hybrid search settings.
"""

import logging
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .exceptions import InvalidConfigValueError

logger = logging.getLogger(__name__)


class RetrievalConfig(BaseModel):
    """
    Configuration for RAG retrieval operations.

    Controls how documents are retrieved from vector databases,
    including search parameters, similarity thresholds, and
    multi-fact retrieval settings.

    Attributes:
        initial_retrieval_k: Initial number of documents to retrieve
        final_retrieval_k: Final number after reranking
        similarity_score_threshold: Minimum similarity score (0.0-1.0)
        guideline_search_k: Number of guideline documents to retrieve
        generated_doc_search_k: Number of generated documents to retrieve
        use_hybrid_multi_fact_approach: Enable multi-fact retrieval
        max_sources_per_fact: Maximum sources per individual fact

    Example:
        >>> config = RetrievalConfig()
        >>> print(config.initial_retrieval_k)
        20
        >>> config.similarity_score_threshold = 0.7
    """

    # General retrieval parameters
    initial_retrieval_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Initial number of documents to retrieve before reranking"
    )

    final_retrieval_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Final number of documents after reranking"
    )

    similarity_score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for document relevance"
    )

    # Domain-specific search parameters
    guideline_search_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of guideline documents to retrieve"
    )

    generated_doc_search_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of generated documents to retrieve"
    )

    # Hybrid multi-fact retrieval
    use_hybrid_multi_fact_approach: bool = Field(
        default=True,
        description="Enable multi-fact retrieval for complex queries"
    )

    max_sources_per_fact: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum sources to retrieve per individual fact"
    )

    model_config = {
        "validate_assignment": True
    }

    @field_validator('final_retrieval_k')
    @classmethod
    def validate_final_k(cls, v: int, info) -> int:
        """Ensure final_k <= initial_k."""
        if 'initial_retrieval_k' in info.data:
            initial_k = info.data['initial_retrieval_k']
            if v > initial_k:
                raise InvalidConfigValueError(
                    f"final_retrieval_k ({v}) cannot be greater than "
                    f"initial_retrieval_k ({initial_k})"
                )
        return v

    @field_validator('similarity_score_threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate similarity threshold is reasonable."""
        if v < 0.3:
            logger.warning(
                f"Very low similarity threshold ({v}). "
                "This may include many irrelevant documents."
            )
        elif v > 0.9:
            logger.warning(
                f"Very high similarity threshold ({v}). "
                "This may exclude relevant documents."
            )
        return v

    def get_retrieval_summary(self) -> str:
        """
        Get a summary of the retrieval configuration.

        Returns:
            Human-readable summary string
        """
        return (
            f"Retrieval: {self.initial_retrieval_k}→{self.final_retrieval_k} docs, "
            f"threshold={self.similarity_score_threshold:.2f}, "
            f"multi-fact={'enabled' if self.use_hybrid_multi_fact_approach else 'disabled'}"
        )

    def log_configuration(self) -> None:
        """Log the current retrieval configuration."""
        logger.info("Retrieval Configuration:")
        logger.info(f"  Initial Retrieval K: {self.initial_retrieval_k}")
        logger.info(f"  Final Retrieval K: {self.final_retrieval_k}")
        logger.info(f"  Similarity Threshold: {self.similarity_score_threshold}")
        logger.info(f"  Guideline Search K: {self.guideline_search_k}")
        logger.info(f"  Generated Doc Search K: {self.generated_doc_search_k}")
        logger.info(
            f"  Multi-Fact Approach: "
            f"{'enabled' if self.use_hybrid_multi_fact_approach else 'disabled'}"
        )
        if self.use_hybrid_multi_fact_approach:
            logger.info(f"  Max Sources Per Fact: {self.max_sources_per_fact}")
