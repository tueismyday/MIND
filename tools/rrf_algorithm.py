"""
Reciprocal Rank Fusion (RRF) algorithm implementation.

This module implements the RRF algorithm for combining multiple ranking
lists into a single fused ranking. RRF is effective for hybrid search
systems that combine semantic and keyword-based retrieval.

Reference:
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    "Reciprocal rank fusion outperforms condorcet and individual rank
    learning methods." SIGIR '09.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from tools.constants import RRF_K_VALUE, RRF_NORMALIZATION_MAX
from tools.scoring import ScoreNormalizer


logger = logging.getLogger(__name__)


@dataclass
class RRFResult:
    """
    Container for RRF fusion results.

    Attributes:
        doc_idx: Document index
        rrf_score: Combined RRF score
        ranking_details: Details from each ranking system
        normalized_score: RRF score normalized to 0-100 scale
    """

    doc_idx: int
    rrf_score: float
    ranking_details: Dict[str, Any] = field(default_factory=dict)
    normalized_score: float = 0.0

    def add_ranking(self, ranking_id: str, rank: int, original_score: float, k: int) -> None:
        """
        Add contribution from a ranking system.

        Args:
            ranking_id: Identifier for the ranking system
            rank: Position in this ranking (1-indexed)
            original_score: Original score from this system
            k: RRF smoothing constant
        """
        rrf_contribution = 1.0 / (k + rank)
        self.rrf_score += rrf_contribution

        self.ranking_details[ranking_id] = {
            'rank': rank,
            'original_score': original_score,
            'rrf_contribution': rrf_contribution
        }


class RRFAlgorithm:
    """
    Reciprocal Rank Fusion algorithm for combining multiple rankings.

    RRF combines rankings using a simple but effective formula:
        score(d) = Σ 1 / (k + rank(d))

    where:
        - d is a document
        - k is a smoothing constant (typically 60)
        - rank(d) is the position of d in each ranking (1-indexed)
        - Σ sums over all ranking systems that include d

    The algorithm is:
        1. Order-invariant (doesn't depend on how rankings are ordered)
        2. Robust to outliers in individual rankings
        3. Simple and parameter-light (only k needs tuning)
        4. Effective at combining heterogeneous retrieval systems

    Example:
        >>> rrf = RRFAlgorithm(k=60)
        >>> semantic_ranking = [(0, 0.9), (1, 0.8), (2, 0.7)]  # (doc_idx, score)
        >>> keyword_ranking = [(1, 5.2), (0, 4.1), (3, 3.5)]
        >>> fused = rrf.fuse([semantic_ranking, keyword_ranking])
        >>> print(fused[0])  # Top result
        (1, 0.0327)  # doc_idx=1, rrf_score=0.0327
    """

    def __init__(self, k: int = RRF_K_VALUE):
        """
        Initialize RRF algorithm.

        Args:
            k: Smoothing constant (typically 60).
               Smaller k gives more weight to top-ranked items.
               Larger k is more conservative.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.k = k
        logger.debug(f"Initialized RRFAlgorithm with k={k}")

    def fuse(
        self,
        rankings: List[List[Tuple[int, float]]],
        rank_window: Optional[int] = None,
        normalize: bool = False
    ) -> List[Tuple[int, float]]:
        """
        Fuse multiple rankings using RRF.

        Args:
            rankings: List of rankings, where each ranking is a list of
                     (doc_idx, score) tuples sorted by relevance
            rank_window: Optional window size to limit rank consideration.
                        Only items within top-N of each ranking contribute.
            normalize: Whether to normalize final scores to 0-100 scale

        Returns:
            List of (doc_idx, rrf_score) tuples sorted by RRF score

        Example:
            >>> rrf = RRFAlgorithm(k=60)
            >>> r1 = [(0, 0.9), (1, 0.7), (2, 0.5)]
            >>> r2 = [(1, 8.0), (0, 6.0), (2, 4.0)]
            >>> fused = rrf.fuse([r1, r2], rank_window=10)
        """
        if not rankings:
            logger.warning("No rankings provided for fusion")
            return []

        logger.debug(
            f"Fusing {len(rankings)} rankings with k={self.k}, "
            f"rank_window={rank_window}, normalize={normalize}"
        )

        # Calculate RRF scores
        rrf_results = self._calculate_rrf_scores(rankings, rank_window)

        # Sort by RRF score (descending)
        sorted_results = sorted(rrf_results.values(), key=lambda x: x.rrf_score, reverse=True)

        # Convert to list of tuples
        result_tuples = [(r.doc_idx, r.rrf_score) for r in sorted_results]

        # Normalize if requested
        if normalize and result_tuples:
            result_tuples = self._normalize_scores(result_tuples)

        logger.debug(
            f"RRF fusion complete: {len(result_tuples)} documents, "
            f"score range [{result_tuples[-1][1]:.4f}, {result_tuples[0][1]:.4f}]"
        )

        return result_tuples

    def fuse_with_details(
        self,
        rankings: List[List[Tuple[int, float]]],
        rank_window: Optional[int] = None,
        normalize: bool = False
    ) -> List[RRFResult]:
        """
        Fuse rankings and return detailed results.

        Args:
            rankings: List of rankings
            rank_window: Optional window size
            normalize: Whether to normalize scores

        Returns:
            List of RRFResult objects with detailed breakdown
        """
        if not rankings:
            return []

        # Calculate RRF scores
        rrf_results = self._calculate_rrf_scores(rankings, rank_window)

        # Sort by RRF score
        sorted_results = sorted(rrf_results.values(), key=lambda x: x.rrf_score, reverse=True)

        # Normalize if requested
        if normalize:
            self._normalize_results(sorted_results)

        return sorted_results

    def _calculate_rrf_scores(
        self,
        rankings: List[List[Tuple[int, float]]],
        rank_window: Optional[int]
    ) -> Dict[int, RRFResult]:
        """
        Calculate RRF scores for all documents.

        Args:
            rankings: List of rankings
            rank_window: Optional rank window

        Returns:
            Dictionary mapping doc_idx to RRFResult
        """
        rrf_results = {}

        # Process each ranking system
        for ranking_id, ranking in enumerate(rankings):
            # Apply rank window if specified
            windowed_ranking = ranking[:rank_window] if rank_window else ranking

            # Process each document in this ranking
            for rank, (doc_idx, original_score) in enumerate(windowed_ranking, 1):
                # Create result entry if doesn't exist
                if doc_idx not in rrf_results:
                    rrf_results[doc_idx] = RRFResult(doc_idx=doc_idx, rrf_score=0.0)

                # Add contribution from this ranking
                rrf_results[doc_idx].add_ranking(
                    ranking_id=f"ranking_{ranking_id}",
                    rank=rank,
                    original_score=original_score,
                    k=self.k
                )

        return rrf_results

    def _normalize_scores(
        self,
        result_tuples: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Normalize RRF scores to 0-100 scale.

        Args:
            result_tuples: List of (doc_idx, rrf_score) tuples

        Returns:
            List of (doc_idx, normalized_score) tuples
        """
        if not result_tuples:
            return []

        doc_indices = [doc_idx for doc_idx, _ in result_tuples]
        scores = np.array([score for _, score in result_tuples])

        normalized_scores = ScoreNormalizer.normalize_to_range(
            scores,
            target_min=0.0,
            target_max=RRF_NORMALIZATION_MAX
        )

        return list(zip(doc_indices, normalized_scores))

    def _normalize_results(self, results: List[RRFResult]) -> None:
        """
        Normalize scores in RRFResult objects (in-place).

        Args:
            results: List of RRFResult objects
        """
        if not results:
            return

        scores = np.array([r.rrf_score for r in results])
        normalized_scores = ScoreNormalizer.normalize_to_range(
            scores,
            target_min=0.0,
            target_max=RRF_NORMALIZATION_MAX
        )

        for result, norm_score in zip(results, normalized_scores):
            result.normalized_score = float(norm_score)

    def explain_score(
        self,
        doc_idx: int,
        rankings: List[List[Tuple[int, float]]],
        rank_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Explain how RRF score was calculated for a specific document.

        Args:
            doc_idx: Document index to explain
            rankings: The rankings used for fusion
            rank_window: Rank window that was applied

        Returns:
            Dictionary with detailed score breakdown

        Example:
            >>> rrf = RRFAlgorithm(k=60)
            >>> rankings = [[(0, 0.9), (1, 0.7)], [(1, 5.0), (0, 3.0)]]
            >>> explanation = rrf.explain_score(doc_idx=1, rankings=rankings)
            >>> print(explanation['total_rrf_score'])
            0.0327
        """
        explanation = {
            'document_index': doc_idx,
            'rrf_k': self.k,
            'rank_window': rank_window,
            'rankings': {},
            'total_rrf_score': 0.0
        }

        for ranking_id, ranking in enumerate(rankings):
            # Apply rank window if specified
            windowed_ranking = ranking[:rank_window] if rank_window else ranking

            # Find document in this ranking
            for rank, (idx, score) in enumerate(windowed_ranking, 1):
                if idx == doc_idx:
                    rrf_contribution = 1.0 / (self.k + rank)
                    explanation['total_rrf_score'] += rrf_contribution

                    explanation['rankings'][f'ranking_{ranking_id}'] = {
                        'rank': rank,
                        'original_score': score,
                        'rrf_contribution': rrf_contribution,
                        'formula': f"1 / ({self.k} + {rank}) = {rrf_contribution:.6f}"
                    }
                    break

        return explanation


# Convenience function
def reciprocal_rank_fusion(
    rankings: List[List[Tuple[int, float]]],
    k: int = RRF_K_VALUE,
    rank_window: Optional[int] = None,
    normalize: bool = False
) -> List[Tuple[int, float]]:
    """
    Convenience function for RRF fusion.

    Args:
        rankings: List of rankings to fuse
        k: RRF smoothing constant
        rank_window: Optional rank window
        normalize: Whether to normalize scores

    Returns:
        Fused ranking as list of (doc_idx, score) tuples
    """
    rrf = RRFAlgorithm(k=k)
    return rrf.fuse(rankings, rank_window=rank_window, normalize=normalize)
