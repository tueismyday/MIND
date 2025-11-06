"""
Scoring utilities for retrieval ranking.

This module provides scoring and normalization utilities for retrieval systems,
including recency calculation, score normalization, and cross-encoder scoring.
"""

import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from utils.text_processing import parse_date_safe
from tools.constants import (
    RECENCY_MAX_BOOST,
    RECENCY_DECAY_CONSTANT,
    MIN_SCORE_RANGE,
    CROSS_ENCODER_MAX_SCORE,
    RRF_NORMALIZATION_MAX
)


logger = logging.getLogger(__name__)


class RecencyCalculator:
    """
    Calculate temporal relevance scores for documents.

    Uses exponential decay to boost scores of recent documents, with the
    decay calculated relative to the most recent document in the corpus
    rather than the current date.

    This approach ensures stable rankings even as time passes, as long as
    the document corpus doesn't change.

    Attributes:
        max_boost: Maximum recency boost score
        decay_constant: Controls decay steepness (smaller = steeper decay)
        most_recent_date: Reference date (most recent document in corpus)
    """

    def __init__(
        self,
        max_boost: float = RECENCY_MAX_BOOST,
        decay_constant: int = RECENCY_DECAY_CONSTANT,
        most_recent_date: Optional[datetime] = None
    ):
        """
        Initialize recency calculator.

        Args:
            max_boost: Maximum boost score for most recent documents
            decay_constant: Exponential decay constant (smaller = steeper)
            most_recent_date: Reference date for recency calculation
        """
        self.max_boost = max_boost
        self.decay_constant = decay_constant
        self.most_recent_date = most_recent_date

        logger.debug(
            f"Initialized RecencyCalculator: max_boost={max_boost}, "
            f"decay_constant={decay_constant}, "
            f"reference_date={most_recent_date.strftime('%Y-%m-%d') if most_recent_date else 'None'}"
        )

    def update_reference_date(self, documents: List[Dict[str, Any]]) -> None:
        """
        Update the reference date based on a corpus of documents.

        Finds the most recent valid date in the document corpus and uses
        it as the reference point for recency calculations.

        Args:
            documents: List of documents with 'date' field in metadata
        """
        most_recent = None

        for doc in documents:
            date_str = doc.get('date', '')
            if not date_str:
                continue

            try:
                doc_date = parse_date_safe(date_str)
                # Only consider past dates (ignore future dates which might be typos)
                if doc_date <= datetime.now():
                    if most_recent is None or doc_date > most_recent:
                        most_recent = doc_date
            except Exception as e:
                logger.debug(f"Failed to parse date '{date_str}': {e}")
                continue

        self.most_recent_date = most_recent

        if most_recent:
            logger.info(
                f"Updated reference date to: {most_recent.strftime('%Y-%m-%d')} "
                f"(found in {len(documents)} documents)"
            )
        else:
            logger.warning(
                f"No valid dates found in {len(documents)} documents - "
                "recency boost will be disabled"
            )

    def calculate_boost(self, date_str: str) -> float:
        """
        Calculate recency boost for a document date.

        Uses exponential decay: boost = max_boost * exp(-days_since_most_recent / decay_constant)

        Args:
            date_str: Date string to calculate boost for

        Returns:
            Recency boost score (0 to max_boost)

        Example:
            >>> calc = RecencyCalculator(max_boost=10.0, decay_constant=3)
            >>> calc.most_recent_date = datetime(2024, 1, 15)
            >>> boost = calc.calculate_boost("2024-01-15")  # Most recent
            >>> print(boost)
            10.0
            >>> boost = calc.calculate_boost("2024-01-12")  # 3 days ago
            >>> print(boost)  # ~3.68 (10 * exp(-3/3))
        """
        if not date_str or not self.most_recent_date:
            return 0.0

        try:
            doc_date = parse_date_safe(date_str)
            now = datetime.now()

            # Don't boost future dates (might be typos or scheduled appointments)
            if doc_date > now:
                logger.debug(f"Future date detected: {date_str}, returning 0 boost")
                return 0.0

            # Calculate days since the most recent document in the corpus
            days_since_most_recent = (self.most_recent_date - doc_date).days

            # If this document is more recent than our reference (shouldn't happen
            # if reference is updated correctly), give it max boost
            if days_since_most_recent < 0:
                logger.debug(
                    f"Document date {date_str} is more recent than reference "
                    f"{self.most_recent_date.strftime('%Y-%m-%d')}, giving max boost"
                )
                return self.max_boost

            # Calculate exponential decay
            recency_boost = self.max_boost * np.exp(
                -days_since_most_recent / self.decay_constant
            )

            # Ensure non-negative
            return max(0.0, recency_boost)

        except Exception as e:
            logger.debug(f"Failed to calculate recency boost for '{date_str}': {e}")
            return 0.0

    def calculate_batch(self, dates: List[str]) -> List[float]:
        """
        Calculate recency boosts for multiple dates.

        Args:
            dates: List of date strings

        Returns:
            List of recency boost scores
        """
        return [self.calculate_boost(date) for date in dates]


class ScoreNormalizer:
    """
    Utility for normalizing scores to specific ranges.

    Provides methods for min-max normalization and score scaling to
    standardize scores from different retrieval systems.
    """

    @staticmethod
    def normalize_to_range(
        scores: np.ndarray,
        target_min: float = 0.0,
        target_max: float = 100.0,
        source_min: Optional[float] = None,
        source_max: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize scores to a target range using min-max normalization.

        Args:
            scores: Array of scores to normalize
            target_min: Minimum value of target range
            target_max: Maximum value of target range
            source_min: Minimum of source range (auto-detected if None)
            source_max: Maximum of source range (auto-detected if None)

        Returns:
            Normalized scores in target range

        Example:
            >>> scores = np.array([0.2, 0.5, 0.8, 1.0])
            >>> normalized = ScoreNormalizer.normalize_to_range(scores, 0, 100)
            >>> print(normalized)
            [0.0, 37.5, 75.0, 100.0]
        """
        if len(scores) == 0:
            return np.array([])

        # Auto-detect source range if not provided
        if source_min is None:
            source_min = scores.min()
        if source_max is None:
            source_max = scores.max()

        score_range = source_max - source_min

        # Handle case where all scores are identical
        if score_range < MIN_SCORE_RANGE:
            logger.debug("All scores identical, returning midpoint of target range")
            return np.full_like(scores, (target_min + target_max) / 2.0, dtype=float)

        # Min-max normalization to target range
        normalized = (scores - source_min) / score_range
        scaled = normalized * (target_max - target_min) + target_min

        return scaled

    @staticmethod
    def normalize_dict_scores(
        score_dict: Dict[Any, float],
        target_min: float = 0.0,
        target_max: float = 100.0
    ) -> Dict[Any, float]:
        """
        Normalize scores in a dictionary.

        Args:
            score_dict: Dictionary mapping keys to scores
            target_min: Minimum value of target range
            target_max: Maximum value of target range

        Returns:
            Dictionary with normalized scores

        Example:
            >>> scores = {'doc1': 0.2, 'doc2': 0.8, 'doc3': 0.5}
            >>> normalized = ScoreNormalizer.normalize_dict_scores(scores, 0, 100)
            >>> print(normalized)
            {'doc1': 0.0, 'doc2': 100.0, 'doc3': 50.0}
        """
        if not score_dict:
            return {}

        keys = list(score_dict.keys())
        values = np.array([score_dict[k] for k in keys])

        normalized_values = ScoreNormalizer.normalize_to_range(
            values, target_min, target_max
        )

        return {k: float(v) for k, v in zip(keys, normalized_values)}


class CrossEncoderScorer:
    """
    Wrapper for cross-encoder scoring with normalization.

    Provides batch scoring and automatic normalization of cross-encoder
    outputs to a standardized range.
    """

    def __init__(self, cross_encoder, max_score: float = CROSS_ENCODER_MAX_SCORE):
        """
        Initialize cross-encoder scorer.

        Args:
            cross_encoder: CrossEncoder model instance
            max_score: Maximum score after normalization
        """
        self.cross_encoder = cross_encoder
        self.max_score = max_score

    def score_pairs(
        self,
        query_doc_pairs: List[Tuple[str, str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> Dict[int, float]:
        """
        Score query-document pairs using cross-encoder.

        Args:
            query_doc_pairs: List of (query, document) tuples
            batch_size: Batch size for scoring
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping pair index to normalized score

        Raises:
            CrossEncoderError: If scoring fails
        """
        if not query_doc_pairs:
            return {}

        try:
            # Get raw scores from cross-encoder
            raw_scores = self.cross_encoder.predict(
                query_doc_pairs,
                batch_size=batch_size,
                show_progress_bar=show_progress
            )

            scores_array = np.array(raw_scores)

            # Normalize to 0-max_score range
            normalized_scores = ScoreNormalizer.normalize_to_range(
                scores_array,
                target_min=0.0,
                target_max=self.max_score
            )

            # Create index-to-score mapping
            result = {i: float(score) for i, score in enumerate(normalized_scores)}

            logger.debug(
                f"Scored {len(query_doc_pairs)} pairs: "
                f"raw range=[{scores_array.min():.3f}, {scores_array.max():.3f}], "
                f"normalized range=[{normalized_scores.min():.1f}, {normalized_scores.max():.1f}]"
            )

            return result

        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}", exc_info=True)
            from tools.exceptions import CrossEncoderError
            raise CrossEncoderError(
                reason=str(e),
                num_pairs=len(query_doc_pairs)
            )
