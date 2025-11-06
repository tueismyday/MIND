"""
Two-stage hybrid search using Reciprocal Rank Fusion (RRF) and cross-encoder reranking.

This module implements a sophisticated retrieval pipeline optimized for Danish
medical text. The search process has two distinct stages:

Stage 1 - Filtering (RRF):
    Combines semantic (embedding-based) and keyword (BM25-based) search using
    Reciprocal Rank Fusion to narrow the document corpus.

Stage 2 - Reranking:
    Applies cross-encoder neural reranking and recency boosting to provide
    fresh, context-aware relevance scores.

Key Classes:
    RRFHybridSearch: Main search orchestrator
    RRFHybridRetriever: ChromaDB adapter for hybrid search

Algorithm Details:
    - RRF Formula: score = Σ 1/(k + rank) for each ranking system
    - Cross-encoder: 0-90 point scale (neural relevance)
    - Recency: 0-10 point scale (temporal relevance)
    - Total: 0-100 maximum score

Scoring:
    - Stage 1: RRF normalizes to 0-100 (used for filtering only)
    - Stage 2: Cross-encoder (0-90) + Recency (0-10) = Total (0-100)
    - Recency boost calculated relative to most recent DB entry

Example:
    >>> from tools.hybrid_search import RRFHybridSearch
    >>> search = RRFHybridSearch(embedding_model, k=60)
    >>> search.index_documents(documents)
    >>> results = search.rrf_search("diabetes behandling", top_k=5)
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi

from config.settings import RERANKER_MODEL_NAME, RERANKER_DEVICE, BATCH_SIZE_RERANK
from core.reranker import get_reranker
from tools.constants import (
    RRF_K_VALUE,
    RRF_DEFAULT_RANK_WINDOW,
    RRF_DEFAULT_RERANK_WINDOW_MIN,
    CROSS_ENCODER_MAX_SCORE,
    RECENCY_MAX_BOOST
)
from tools.exceptions import (
    EmptyCorpusError,
    InvalidSearchParametersError,
    IndexingError,
    CrossEncoderError
)
from tools.tokenizer import MedicalTextTokenizer
from tools.rrf_algorithm import RRFAlgorithm
from tools.scoring import RecencyCalculator, CrossEncoderScorer


logger = logging.getLogger(__name__)


class RRFHybridSearch:
    """
    In-memory hybrid search using Reciprocal Rank Fusion (RRF).

    Combines semantic and keyword search with two-stage ranking:
    1. RRF filtering to narrow corpus
    2. Cross-encoder + recency for final scoring

    Attributes:
        embedding_model: Sentence transformer for semantic embeddings
        k: RRF smoothing constant (typically 60)
        documents: Indexed document corpus
        embeddings: Semantic embeddings matrix
        bm25: BM25 keyword search index
        tokenizer: Medical text tokenizer
        rrf_algorithm: RRF fusion algorithm
        recency_calculator: Temporal relevance calculator
        cross_encoder_scorer: Neural reranking scorer
    """

    def __init__(self, embedding_model, k: int = RRF_K_VALUE):
        """
        Initialize RRF hybrid search.

        Args:
            embedding_model: Sentence transformer for semantic embeddings
            k: RRF smoothing constant (typically 60)

        Example:
            >>> from sentence_transformers import SentenceTransformer
            >>> model = SentenceTransformer('model-name')
            >>> search = RRFHybridSearch(model, k=60)
        """
        self.embedding_model = embedding_model
        self.k = k
        self.documents = []
        self.embeddings = None
        self.bm25 = None

        # Initialize components
        self.tokenizer = MedicalTextTokenizer()
        self.rrf_algorithm = RRFAlgorithm(k=k)
        self.recency_calculator = RecencyCalculator(
            max_boost=RECENCY_MAX_BOOST
        )

        # Load cross-encoder
        logger.info(f"Loading cross-encoder: {RERANKER_MODEL_NAME}")
        logger.info(f"Target device: {RERANKER_DEVICE}")

        cross_encoder = self._load_cross_encoder()
        self.cross_encoder_scorer = CrossEncoderScorer(
            cross_encoder,
            max_score=CROSS_ENCODER_MAX_SCORE
        )

        logger.info("RRFHybridSearch initialized successfully")

    def _load_cross_encoder(self):
        """
        Load cross-encoder model with GPU/CPU fallback handling.

        Delegates to get_reranker() from core.reranker which handles all
        loading logic and singleton caching.

        Returns:
            CrossEncoder: Loaded cross-encoder model

        Raises:
            ModelLoadingError: If model loading fails
        """
        try:
            return get_reranker()
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}", exc_info=True)
            from tools.exceptions import ModelLoadingError
            raise ModelLoadingError(
                model_name=RERANKER_MODEL_NAME,
                reason=str(e)
            )

    def index_documents(self, documents: List[Dict]) -> None:
        """
        Index documents for RRF hybrid search.

        Args:
            documents: List of document dictionaries with 'content' and metadata

        Raises:
            IndexingError: If indexing fails
            InvalidSearchParametersError: If documents are invalid

        Example:
            >>> docs = [
            ...     {'content': 'Patient har diabetes', 'date': '2024-01-15'},
            ...     {'content': 'Blodsukker måling 8.5', 'date': '2024-01-14'}
            ... ]
            >>> search.index_documents(docs)
        """
        if not documents:
            raise InvalidSearchParametersError(
                parameter="documents",
                value=documents,
                reason="Document list cannot be empty"
            )

        logger.info(f"Indexing {len(documents)} documents for RRF hybrid search")

        try:
            self.documents = documents

            # Update recency calculator with document dates
            self.recency_calculator.update_reference_date(documents)

            # Prepare content for indexing
            contents = [doc['content'] for doc in documents]

            # Create semantic embeddings
            logger.info("Creating semantic embeddings...")
            self.embeddings = self.embedding_model.encode(contents)
            logger.debug(f"Embeddings shape: {self.embeddings.shape}")

            # Prepare BM25 index
            logger.info("Creating BM25 keyword index...")
            tokenized_docs = self.tokenizer.tokenize_batch(contents)
            self.bm25 = BM25Okapi(tokenized_docs)

            logger.info(
                f"RRF hybrid search index ready (k={self.k}, "
                f"documents={len(documents)})"
            )

            if self.recency_calculator.most_recent_date:
                logger.info(
                    f"Most recent entry date: "
                    f"{self.recency_calculator.most_recent_date.strftime('%Y-%m-%d')}"
                )

        except Exception as e:
            logger.error(f"Document indexing failed: {e}", exc_info=True)
            raise IndexingError(
                reason=str(e),
                num_documents=len(documents)
            )

    def _get_semantic_ranking(
        self,
        query: str,
        window_size: Optional[int] = None,
        note_types: Optional[List[str]] = None
    ) -> List[Tuple[int, float]]:
        """
        Get semantic search ranking using cosine similarity.

        Args:
            query: Search query
            window_size: Maximum number of results to consider
            note_types: Optional filter by note types

        Returns:
            List of (document_index, similarity_score) tuples, sorted by relevance
        """
        if self.embeddings is None:
            return []

        query_embedding = self.embedding_model.encode([query])
        semantic_scores = np.dot(self.embeddings, query_embedding.T).flatten()

        # Create ranked list with filtering
        ranked_results = []
        for i, score in enumerate(semantic_scores):
            # Apply note type filter if specified
            if note_types:
                doc_note_type = self.documents[i].get('entry_type', '')
                if doc_note_type not in note_types:
                    continue

            ranked_results.append((i, score))

        # Sort by score (descending)
        ranked_results.sort(key=lambda x: x[1], reverse=True)

        # Apply window size if specified
        if window_size:
            ranked_results = ranked_results[:window_size]

        logger.debug(
            f"Semantic ranking: {len(ranked_results)} results, "
            f"top score={ranked_results[0][1]:.3f}" if ranked_results else "empty"
        )

        return ranked_results

    def _get_keyword_ranking(
        self,
        query: str,
        window_size: Optional[int] = None,
        note_types: Optional[List[str]] = None
    ) -> List[Tuple[int, float]]:
        """
        Get keyword search ranking using BM25.

        Args:
            query: Search query
            window_size: Maximum number of results to consider
            note_types: Optional filter by note types

        Returns:
            List of (document_index, bm25_score) tuples, sorted by relevance
        """
        if self.bm25 is None:
            return []

        tokenized_query = self.tokenizer.tokenize(query)
        keyword_scores = self.bm25.get_scores(tokenized_query)

        # Create ranked list with filtering
        ranked_results = []
        for i, score in enumerate(keyword_scores):
            # Apply note type filter if specified
            if note_types:
                doc_note_type = self.documents[i].get('entry_type', '')
                if doc_note_type not in note_types:
                    continue

            ranked_results.append((i, score))

        # Sort by score (descending)
        ranked_results.sort(key=lambda x: x[1], reverse=True)

        # Apply window size if specified
        if window_size:
            ranked_results = ranked_results[:window_size]

        logger.debug(
            f"Keyword ranking: {len(ranked_results)} results, "
            f"top score={ranked_results[0][1]:.3f}" if ranked_results else "empty"
        )

        return ranked_results

    def rrf_search(
        self,
        query: str,
        top_k: int = 15,
        rank_window: int = RRF_DEFAULT_RANK_WINDOW,
        semantic_window: Optional[int] = None,
        keyword_window: Optional[int] = None,
        enable_boosting: bool = True,
        note_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Perform hybrid search using TWO-STAGE approach.

        Stage 1 (Filtering): RRF narrows documents from full corpus to top candidates
        Stage 2 (Ranking): Cross-encoder + recency provide fresh scoring of candidates

        Args:
            query: Search query
            top_k: Number of final results to return
            rank_window: Window size for RRF calculation
            semantic_window: Window size for semantic ranking (None = use all)
            keyword_window: Window size for keyword ranking (None = use all)
            enable_boosting: Whether to apply cross-encoder and recency scoring
            note_types: Specific note types to include

        Returns:
            List of search results with detailed ranking information

        Raises:
            EmptyCorpusError: If no documents are indexed
            InvalidSearchParametersError: If parameters are invalid

        Two-Stage Scoring:
            Stage 1 - RRF Filtering: Narrows corpus using semantic + keyword fusion
            Stage 2 - Fresh Scoring (when enable_boosting=True):
                - Cross-encoder: 0-90 (neural relevance scoring)
                - Recency: 0-10 (temporal relevance)
                - Total: 0-100 max

            If enable_boosting=False: Falls back to RRF scores (0-100)

        Example:
            >>> results = search.rrf_search("diabetes behandling", top_k=5)
            >>> for result in results:
            ...     print(f"{result['score']:.1f}: {result['content'][:50]}")
        """
        # Validate parameters
        if top_k <= 0:
            raise InvalidSearchParametersError(
                parameter="top_k",
                value=top_k,
                reason="top_k must be positive"
            )

        if not self.documents or self.embeddings is None or self.bm25 is None:
            raise EmptyCorpusError("hybrid search index")

        logger.info(
            f"RRF search: query='{query}', top_k={top_k}, rank_window={rank_window}, "
            f"note_types={note_types}"
        )

        # Stage 1 - RRF Filtering
        results = self._stage1_filter(
            query, rank_window, semantic_window, keyword_window, note_types
        )

        if not results:
            logger.warning("Stage 1 filtering returned no results")
            return []

        # Stage 2 - Reranking (if enabled)
        if enable_boosting:
            results = self._stage2_rerank(query, results)

        # Sort by final score and limit to top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        final_results = results[:top_k]

        logger.info(
            f"RRF search complete: {len(final_results)} results returned, "
            f"top score={final_results[0]['score']:.1f}"
        )

        return final_results

    def _stage1_filter(
        self,
        query: str,
        rank_window: int,
        semantic_window: Optional[int],
        keyword_window: Optional[int],
        note_types: Optional[List[str]]
    ) -> List[Dict]:
        """
        Stage 1: Filter documents using RRF fusion.

        Args:
            query: Search query
            rank_window: RRF rank window
            semantic_window: Semantic ranking window
            keyword_window: Keyword ranking window
            note_types: Note type filter

        Returns:
            List of filtered results with RRF scores
        """
        logger.debug("Stage 1: RRF filtering")

        # Get individual rankings
        semantic_ranking = self._get_semantic_ranking(query, semantic_window, note_types)
        keyword_ranking = self._get_keyword_ranking(query, keyword_window, note_types)

        logger.info(
            f"Rankings - Semantic: {len(semantic_ranking)}, "
            f"Keyword: {len(keyword_ranking)}"
        )

        # Apply RRF to combine rankings
        rrf_ranking = self.rrf_algorithm.fuse(
            [semantic_ranking, keyword_ranking],
            rank_window=rank_window,
            normalize=True  # Normalize to 0-100
        )

        logger.info(f"RRF fusion: {len(rrf_ranking)} results")

        # Build result list
        results = []
        for doc_idx, rrf_score in rrf_ranking:
            doc = self.documents[doc_idx]

            # Get original scores and ranks
            semantic_score = next((s for i, s in semantic_ranking if i == doc_idx), 0.0)
            keyword_score = next((s for i, s in keyword_ranking if i == doc_idx), 0.0)
            semantic_rank = next((r for r, (i, _) in enumerate(semantic_ranking, 1) if i == doc_idx), None)
            keyword_rank = next((r for r, (i, _) in enumerate(keyword_ranking, 1) if i == doc_idx), None)

            result = {
                'content': doc.get('content', ''),
                'entry_type': doc.get('entry_type', ''),
                'date': doc.get('date', ''),
                'document_id': doc.get('document_id', ''),
                'chunk_index': doc.get('chunk_index', 0),
                'doc_idx': doc_idx,  # Internal index for Stage 2

                'score': rrf_score,  # Will be replaced in Stage 2
                'rrf_score': rrf_score,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'semantic_rank': semantic_rank,
                'keyword_rank': keyword_rank,

                'rrf_k': self.k,
                'rank_window': rank_window
            }

            results.append(result)

        return results

    def _stage2_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Stage 2: Rerank using cross-encoder and recency.

        Args:
            query: Search query
            results: Results from Stage 1

        Returns:
            Results with updated scores
        """
        logger.debug("Stage 2: Cross-encoder + recency reranking")

        # Determine reranking window
        rerank_window = self._calculate_rerank_window(len(results))
        top_candidates = results[:rerank_window]

        logger.info(
            f"Applying cross-encoder reranking to top {rerank_window} candidates "
            f"({rerank_window}/{len(results)})"
        )

        # Prepare query-document pairs for cross-encoder
        query_doc_pairs = [
            (query, result['content'])
            for result in top_candidates
        ]

        # Score with cross-encoder
        try:
            ce_scores = self.cross_encoder_scorer.score_pairs(
                query_doc_pairs,
                batch_size=BATCH_SIZE_RERANK,
                show_progress=True
            )
            logger.info("Cross-encoder scoring complete")
        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}", exc_info=True)
            # Fallback: keep RRF scores
            return results

        # Update scores with cross-encoder + recency
        for i, result in enumerate(top_candidates):
            # Get cross-encoder score (0-90)
            ce_score = ce_scores.get(i, 0.0)

            # Calculate recency boost (0-10)
            recency_boost = self.recency_calculator.calculate_boost(
                result.get('date', '')
            )

            # Final score: cross-encoder + recency
            final_score = ce_score + recency_boost

            # Update result
            result['score'] = final_score
            result['cross_encoder_score'] = ce_score
            result['recency_boost'] = recency_boost

        # Update non-reranked results (keep RRF score as final score)
        for result in results[rerank_window:]:
            result['cross_encoder_score'] = 0.0
            result['recency_boost'] = 0.0
            # score already set to rrf_score from Stage 1

        return results

    def _calculate_rerank_window(self, num_results: int) -> int:
        """
        Calculate reranking window size.

        Args:
            num_results: Total number of results

        Returns:
            Window size for reranking
        """
        if num_results < RRF_DEFAULT_RERANK_WINDOW_MIN:
            return num_results

        return max(RRF_DEFAULT_RERANK_WINDOW_MIN, num_results // 2)

    def explain_rrf_score(
        self,
        doc_idx: int,
        query: str,
        rank_window: int = RRF_DEFAULT_RANK_WINDOW
    ) -> Dict:
        """
        Explain how the RRF score was calculated for a specific document.

        Args:
            doc_idx: Document index
            query: Original query
            rank_window: Rank window used in RRF

        Returns:
            Dictionary with detailed score breakdown

        Example:
            >>> explanation = search.explain_rrf_score(0, "diabetes", 100)
            >>> print(explanation['total_rrf_score'])
        """
        semantic_ranking = self._get_semantic_ranking(query)
        keyword_ranking = self._get_keyword_ranking(query)

        return self.rrf_algorithm.explain_score(
            doc_idx,
            [semantic_ranking, keyword_ranking],
            rank_window
        )


class RRFHybridRetriever:
    """
    Retriever adapter for the RRF hybrid search system.

    Provides a ChromaDB-compatible interface for RRF hybrid search,
    with convenient methods for retrieval and result formatting.

    Attributes:
        chroma_db: ChromaDB instance
        rrf_search: RRF hybrid search instance
    """

    def __init__(self, chroma_db, embedding_model, rrf_k: int = RRF_K_VALUE):
        """
        Initialize RRF hybrid retriever.

        Args:
            chroma_db: ChromaDB instance
            embedding_model: Sentence transformer model
            rrf_k: RRF smoothing constant

        Example:
            >>> from core.database import db_manager
            >>> from sentence_transformers import SentenceTransformer
            >>> model = SentenceTransformer('model-name')
            >>> retriever = RRFHybridRetriever(db_manager.patient_db, model)
        """
        self.chroma_db = chroma_db
        self.rrf_search = RRFHybridSearch(embedding_model, k=rrf_k)
        self._initialize_from_chroma()

    def _initialize_from_chroma(self) -> None:
        """
        Initialize RRF search index from existing Chroma database.

        Raises:
            IndexingError: If initialization fails
        """
        logger.info("Initializing RRF hybrid search from Chroma")

        try:
            all_docs = self.chroma_db.get()

            documents = []
            for i, (doc_id, content, metadata) in enumerate(zip(
                all_docs['ids'],
                all_docs['documents'],
                all_docs['metadatas']
            )):
                doc = {
                    'content': content,
                    'document_id': doc_id,
                    'chunk_index': i,
                    'entry_type': metadata.get('entry_type', ''),
                    'date': metadata.get('date', ''),
                }
                documents.append(doc)

            # Index documents
            self.rrf_search.index_documents(documents)

            logger.info(
                f"RRF hybrid search ready with {len(documents)} documents"
            )

        except Exception as e:
            logger.error(f"Failed to initialize RRF search: {e}", exc_info=True)
            raise IndexingError(
                reason=f"ChromaDB initialization failed: {e}",
                num_documents=0
            )

    def retrieve_with_sources(
        self,
        query: str,
        max_sources: int = 3,
        rank_window: int = RRF_DEFAULT_RANK_WINDOW,
        enable_explanation: bool = False,
        note_types: Optional[List[str]] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Retrieve patient information using RRF hybrid search.

        Args:
            query: Search query
            max_sources: Maximum number of results to return
            rank_window: RRF rank window size
            enable_explanation: Whether to include RRF score explanations
            note_types: Optional note type filter

        Returns:
            Tuple of (formatted_content, sources_list)

        Example:
            >>> content, sources = retriever.retrieve_with_sources(
            ...     "diabetes behandling", max_sources=5
            ... )
        """
        # Perform RRF search
        results = self.rrf_search.rrf_search(
            query,
            top_k=max_sources * 2,  # Get more results for better selection
            rank_window=rank_window,
            note_types=note_types
        )

        # Format results
        content, sources = self._format_results(
            results[:max_sources],
            query,
            rank_window,
            enable_explanation
        )

        return content, sources

    def _format_results(
        self,
        results: List[Dict],
        query: str,
        rank_window: int,
        enable_explanation: bool
    ) -> Tuple[str, List[Dict]]:
        """
        Format search results as markdown content.

        Args:
            results: Search results
            query: Original query
            rank_window: Rank window used
            enable_explanation: Include explanations

        Returns:
            Tuple of (formatted_content, sources_list)
        """
        sources = []
        content_parts = [
            "# Patientoplysninger (RRF Hybrid Search - Two-Stage Approach)\n",
            "*Stage 1: RRF filtering | Stage 2: Cross-encoder + Recency scoring*\n",
            f"*Score Scale: Cross-Encoder (0-{CROSS_ENCODER_MAX_SCORE:.0f}) + "
            f"Recency (0-{RECENCY_MAX_BOOST:.0f}) = Total (0-100)*\n"
        ]

        for i, result in enumerate(results):
            # Calculate relevance percentage
            max_score = results[0]['score'] if results else 1
            relevance = int((result['score'] / max_score) * 100) if max_score > 0 else 0

            # Build source reference
            source_ref = {
                'timestamp': result['date'],
                'entry_type': result['entry_type'],
                'relevance': relevance,
                'snippet': result['content'][:150],
                'full_content': result['content'],

                # RRF-specific details
                'rrf_score': result['rrf_score'],
                'semantic_score': result['semantic_score'],
                'keyword_score': result['keyword_score'],
                'semantic_rank': result['semantic_rank'],
                'keyword_rank': result['keyword_rank'],
                'cross_encoder_score': result.get('cross_encoder_score', 0.0),
                'recency_boost': result.get('recency_boost', 0.0),
                'rrf_k': result['rrf_k']
            }
            sources.append(source_ref)

            # Add to content display
            content_parts.append(
                f"## [{i+1}] {result['entry_type']} ({result['date']})"
            )

            # Detailed scoring information
            score_info = [
                f"**Total: {result['score']:.1f}/100**",
                f"CE: {result.get('cross_encoder_score', 0.0):.1f}/{CROSS_ENCODER_MAX_SCORE:.0f}",
                f"Recency: {result.get('recency_boost', 0.0):.1f}/{RECENCY_MAX_BOOST:.0f}",
                f"[RRF Filter: {result['rrf_score']:.1f}/100]"
            ]

            rank_info = [
                f"Sem: #{result['semantic_rank'] or 'N/A'}",
                f"Key: #{result['keyword_rank'] or 'N/A'}"
            ]

            content_parts.append(f"*{' | '.join(score_info)}*")
            content_parts.append(f"*{' | '.join(rank_info)} | Relevans: {relevance}%*\n")

            # Optional RRF explanation
            if enable_explanation and i < 2:
                explanation = self.rrf_search.explain_rrf_score(
                    result['doc_idx'], query, rank_window
                )
                content_parts.append("**RRF Score Breakdown:**")
                for ranking_type, details in explanation['rankings'].items():
                    content_parts.append(
                        f"- {ranking_type.replace('_', ' ').title()}: "
                        f"Rank #{details['rank']} → {details['formula']}"
                    )
                content_parts.append("")

            content_parts.append(result['content'])
            content_parts.append("\n---\n")

        full_content = "\n".join(content_parts)
        return full_content, sources
