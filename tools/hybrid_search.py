"""
Enhanced lightweight in-memory hybrid search implementation with Reciprocal Rank Fusion (RRF).
Combines semantic search with BM25 keyword search using RRF instead of simple weighted combination.
Uses singleton pattern to prevent loading the same reranker model multiple times.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re
from datetime import datetime

from config.settings import RERANKER_MODEL_NAME, RERANKER_DEVICE
from utils.text_processing import parse_date_safe
import torch

# Global cache for cross-encoder/reranker model (singleton pattern)
_cross_encoder_cache = None
_cross_encoder_device_cache = None

class RRFHybridSearch:
    """
    In-memory hybrid search using Reciprocal Rank Fusion (RRF) to combine semantic and keyword search.
    """

    def __init__(self, embedding_model, k: int = 60):
        """
        Initialize RRF hybrid search.

        Args:
            embedding_model: Sentence transformer for semantic embeddings
            k: RRF smoothing constant (typically 60)
        """
        self.embedding_model = embedding_model
        self.k = k  # RRF smoothing constant
        self.documents = []
        self.embeddings = None
        self.bm25 = None

        print(f"[INFO] Loading cross-encoder: {RERANKER_MODEL_NAME}")
        print(f"[INFO] Target device: {RERANKER_DEVICE}")

        # Load cross-encoder with fallback handling
        self.cross_encoder = self._load_cross_encoder()
        print(f"[SUCCESS] Cross-encoder loaded successfully")

    def _load_cross_encoder(self):
        """
        Load cross-encoder model with GPU/CPU fallback handling.

        Uses torch.cuda.mem_get_info() to accurately detect free GPU memory,
        accounting for vLLM or other GPU usage. Automatically falls back to
        CPU if insufficient memory is available.

        Uses singleton pattern - only loads the model once and returns cached
        instance on subsequent calls.

        Returns:
            CrossEncoder: Loaded cross-encoder model
        """
        global _cross_encoder_cache, _cross_encoder_device_cache

        # Return cached model if already loaded
        if _cross_encoder_cache is not None:
            print(f"[INFO] Reusing cached cross-encoder model (device: {_cross_encoder_device_cache})")
            return _cross_encoder_cache

        device_attempts = []

        if RERANKER_DEVICE == "cuda" or RERANKER_DEVICE.startswith("cuda:"):
            if torch.cuda.is_available():
                device_attempts = [RERANKER_DEVICE, "cpu"]
            else:
                print(f"[WARNING] CUDA requested but not available, using CPU")
                device_attempts = ["cpu"]
        else:
            device_attempts = [RERANKER_DEVICE]

        last_error = None

        for device in device_attempts:
            try:
                print(f"[INFO] Attempting to load cross-encoder on {device}...")

                # Check GPU memory if using CUDA
                if device.startswith("cuda"):
                    gpu_idx = 0 if device == "cuda" else int(device.split(":")[1])

                    # Use mem_get_info() for ACTUAL free memory (includes vLLM usage)
                    free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(gpu_idx)
                    free_mem = free_mem_bytes / (1024**3)
                    total_mem = total_mem_bytes / (1024**3)
                    used_mem = total_mem - free_mem

                    print(f"[RERANKER GPU] Total: {total_mem:.2f}GB, Used: {used_mem:.2f}GB, Free: {free_mem:.2f}GB")

                    # Need at least 1.5GB free for reranker model (0.6B params ≈ 1.2GB + buffer)
                    MIN_FREE_FOR_RERANKER = 1.5

                    if free_mem < MIN_FREE_FOR_RERANKER:
                        print(f"[RERANKER GPU] Insufficient free memory ({free_mem:.2f}GB < {MIN_FREE_FOR_RERANKER}GB)")
                        print(f"[RERANKER GPU] Falling back to CPU")
                        continue  # Skip to CPU

                cross_encoder = CrossEncoder(RERANKER_MODEL_NAME, device=device)

                print(f"[SUCCESS] Cross-encoder loaded on {device}")

                # Cache the model for future use (singleton pattern)
                _cross_encoder_cache = cross_encoder
                _cross_encoder_device_cache = device
                print(f"[INFO] Cross-encoder model cached for reuse")

                return cross_encoder

            except torch.cuda.OutOfMemoryError as e:
                print(f"[WARNING] GPU out of memory: {str(e)}")
                if device != "cpu":
                    print(f"[INFO] Falling back to CPU...")
                    last_error = e
                    continue
                else:
                    raise Exception("Out of memory even on CPU - insufficient system resources")

            except Exception as e:
                error_msg = str(e).lower()
                print(f"[WARNING] Failed to load on {device}: {str(e)}")

                # Check if it's a GPU-related error
                if any(keyword in error_msg for keyword in ['cuda', 'gpu', 'memory', 'device']):
                    if device != "cpu":
                        print(f"[INFO] GPU error detected, falling back to CPU...")
                        last_error = e
                        continue

                last_error = e

                if device == device_attempts[-1]:
                    raise Exception(f"Failed to load cross-encoder on any device. Last error: {str(e)}")

        raise Exception(f"Failed to load cross-encoder. Last error: {str(last_error)}")
        
        # Danish stopwords for better keyword matching
        self.danish_stopwords = {
            'og', 'i', 'jeg', 'det', 'at', 'en', 'den', 'til', 'er', 'som', 
            'på', 'de', 'med', 'han', 'af', 'for', 'ikke', 'der', 'var', 'mig',
            'sig', 'men', 'et', 'har', 'om', 'vi', 'min', 'havde', 'ham', 'hun',
            'nu', 'over', 'da', 'fra', 'du', 'ud', 'sin', 'dem', 'os', 'op',
            'man', 'hans', 'hvor', 'eller', 'hvad', 'skal', 'selv', 'her',
            'alle', 'vil', 'blev', 'kunne', 'ind', 'når', 'være', 'dog', 'noget',
            'havde', 'mod', 'disse', 'hvis', 'din', 'nogle', 'hos', 'blive',
            'mange', 'ad', 'bliver', 'hendes', 'været', 'thi', 'jer', 'sådan'
        }
    
    def index_documents(self, documents: List[Dict]):
        """
        Index documents for RRF hybrid search.
        
        Args:
            documents: List of document dictionaries with 'content' and metadata
        """
        
        self.documents = documents
        
        # Prepare content for indexing
        contents = [doc['content'] for doc in documents]
        
        print(f"[INFO] Indexing {len(contents)} documents for RRF hybrid search...")
        
        # Create semantic embeddings
        print("[INFO] Creating semantic embeddings...")
        self.embeddings = self.embedding_model.encode(contents)
        
        # Prepare BM25 index
        print("[INFO] Creating BM25 keyword index...")
        tokenized_docs = [self._tokenize_danish_medical(content) for content in contents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"[INFO] RRF hybrid search index ready (k={self.k})")
    
    def _tokenize_danish_medical(self, text: str) -> List[str]:
        """
        Tokenize text for Danish medical content.
        Preserves medical terms, numbers, and dosages.
        """
        
        # Convert to lowercase
        text = text.lower()
        
        # Preserve medical patterns (dosages, measurements, etc.)
        # This regex preserves: numbers with units, medical abbreviations, etc.
        tokens = re.findall(r'\b(?:\d+(?:[.,]\d+)?(?:mg|g|ml|l|%|mm|cm|kg|år|måned|dag|timer|min)?\b|\w+)\b', text)
        
        # Remove Danish stopwords but keep medical terms
        filtered_tokens = []
        for token in tokens:
            # Keep medical/clinical terms even if they might be stopwords in other contexts
            if (token not in self.danish_stopwords or 
                len(token) > 4 or  # Keep longer words
                token.isdigit() or  # Keep numbers
                any(char.isdigit() for char in token)):  # Keep tokens with numbers
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def _get_semantic_ranking(self, query: str, window_size: int = None, note_types: List[str] = None) -> List[Tuple[int, float]]:
        """
        Get semantic search ranking using cosine similarity.
        
        Args:
            query: Search query
            window_size: Maximum number of results to consider for ranking
            
        Returns:
            List of (document_index, similarity_score) tuples, sorted by relevance
        """
        if self.embeddings is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        semantic_scores = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Create ranked list (document_index, score)
        ranked_results = []
        for i, score in enumerate(semantic_scores):
            # Apply note type filter
            if note_types:
                doc_note_type = self.documents[i].get('entry_type', '')
                if doc_note_type not in note_types:
                    continue  # Skip documents that don't match note types
            
            ranked_results.append((i, score))
        
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply window size if specified
        if window_size:
            ranked_results = ranked_results[:window_size]
        
        return ranked_results
    
    def _get_keyword_ranking(self, query: str, window_size: int = None, note_types: List[str] = None) -> List[Tuple[int, float]]:
        """
        Get keyword search ranking using BM25.
        
        Args:
            query: Search query
            window_size: Maximum number of results to consider for ranking
            
        Returns:
            List of (document_index, bm25_score) tuples, sorted by relevance
        """
        if self.bm25 is None:
            return []
        
        tokenized_query = self._tokenize_danish_medical(query)
        keyword_scores = self.bm25.get_scores(tokenized_query)
        
        # Create ranked list (document_index, score)
        ranked_results = []
        for i, score in enumerate(keyword_scores):
            # Apply note type filter
            if note_types:
                doc_note_type = self.documents[i].get('entry_type', '')
                if doc_note_type not in note_types:
                    continue  # Skip documents that don't match note types
            
            ranked_results.append((i, score))
            
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply window size if specified
        if window_size:
            ranked_results = ranked_results[:window_size]
        
        return ranked_results
    
    def _apply_reciprocal_rank_fusion(self, rankings: List[List[Tuple[int, float]]], 
                                    rank_window: int = None) -> List[Tuple[int, float]]:
        """
        Apply Reciprocal Rank Fusion to combine multiple rankings.
        
        Args:
            rankings: List of rankings, where each ranking is [(doc_idx, score), ...]
            rank_window: Window size for RRF calculation (limits rank consideration)
            
        Returns:
            List of (document_index, rrf_score) tuples, sorted by RRF score
        """
        rrf_scores = {}
        
        # For each ranking system
        for ranking in rankings:
            # Apply rank window if specified
            window_ranking = ranking[:rank_window] if rank_window else ranking
            
            # For each document in this ranking
            for rank, (doc_idx, original_score) in enumerate(window_ranking, 1):
                if doc_idx not in rrf_scores:
                    rrf_scores[doc_idx] = {
                        'rrf_score': 0.0,
                        'ranking_details': {},
                        'doc_idx': doc_idx
                    }
                
                # Calculate RRF contribution: 1 / (k + rank)
                rrf_contribution = 1.0 / (self.k + rank)
                rrf_scores[doc_idx]['rrf_score'] += rrf_contribution
                
                # Store ranking details for transparency
                ranking_id = len(rrf_scores[doc_idx]['ranking_details'])
                rrf_scores[doc_idx]['ranking_details'][f'ranking_{ranking_id}'] = {
                    'rank': rank,
                    'original_score': original_score,
                    'rrf_contribution': rrf_contribution
                }
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        
        return [(result['doc_idx'], result['rrf_score']) for result in sorted_results]
    
    def rrf_search(self, query: str, top_k: int = 15, 
                   rank_window: int = 100,
                   semantic_window: int = None,
                   keyword_window: int = None,
                   enable_boosting: bool = True,
                    note_types: List[str] = None) -> List[Dict]:
        """
        Perform hybrid search using Reciprocal Rank Fusion with cross-encoder reranking.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            rank_window: Window size for RRF calculation (default: 150)
            semantic_window: Window size for semantic ranking (None = use all)
            keyword_window: Window size for keyword ranking (None = use all)
            enable_boosting: Whether to apply cross-encoder and recency boosting
            
        Returns:
            List of search results with RRF scores and detailed ranking information
        """
        
        if not self.documents or self.embeddings is None or self.bm25 is None:
            return []
        
        print(f"[INFO] Performing RRF search (k={self.k}, rank_window={rank_window})")
        
        # Get individual rankings
        semantic_ranking = self._get_semantic_ranking(query, semantic_window, note_types)
        keyword_ranking = self._get_keyword_ranking(query, keyword_window, note_types)
        
        print(f"[INFO] Semantic ranking: {len(semantic_ranking)} results")
        print(f"[INFO] Keyword ranking: {len(keyword_ranking)} results")
        
        # Apply RRF to combine rankings
        rrf_ranking = self._apply_reciprocal_rank_fusion(
            [semantic_ranking, keyword_ranking], 
            rank_window
        )
        
        print(f"[INFO] RRF fusion: {len(rrf_ranking)} results")
        
        rerank_window = min(len(rrf_ranking), top_k * 3)
        top_candidates = rrf_ranking[:rerank_window]
        
        print(f"[INFO] Applying cross-encoder reranking to top {rerank_window} candidates")
        
        cross_encoder_scores = {}
        if enable_boosting and top_candidates:
            candidates_for_scoring = [
                (doc_idx, self.documents[doc_idx]) 
                for doc_idx, _ in top_candidates
            ]
            cross_encoder_scores = self._calculate_cross_encoder_scores(query, candidates_for_scoring)
            print(f"[INFO] Cross-encoder scoring complete")
        
        results = []
        for doc_idx, rrf_score in top_candidates:
            doc = self.documents[doc_idx]
            
            semantic_score = next((score for idx, score in semantic_ranking if idx == doc_idx), 0.0)
            keyword_score = next((score for idx, score in keyword_ranking if idx == doc_idx), 0.0)
            
            semantic_rank = next((rank for rank, (idx, _) in enumerate(semantic_ranking, 1) if idx == doc_idx), None)
            keyword_rank = next((rank for rank, (idx, _) in enumerate(keyword_ranking, 1) if idx == doc_idx), None)
            
            final_score = rrf_score
            
            cross_encoder_score = 0.0
            recency_boost = 0.0
            
            if enable_boosting:
                cross_encoder_score = cross_encoder_scores.get(doc_idx, 0.0)
                final_score += cross_encoder_score
                
                recency_boost = self._calculate_recency_boost(doc.get('date', ''))
                final_score += recency_boost
            
            result = {
                'content': doc.get('content', ''),
                'entry_type': doc.get('entry_type', ''),
                'date': doc.get('date', ''),
                'document_id': doc.get('document_id', ''),
                'chunk_index': doc.get('chunk_index', 0),

                'score': final_score,
                'rrf_score': rrf_score,

                'semantic_score': semantic_score,
                'keyword_score': keyword_score,

                'semantic_rank': semantic_rank,
                'keyword_rank': keyword_rank,

                'cross_encoder_score': cross_encoder_score,
                'recency_boost': recency_boost,

                'rrf_k': self.k,
                'rank_window': rank_window
            }
            
            results.append(result)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _calculate_cross_encoder_scores(self, query: str, documents: List[Tuple[int, Dict]]) -> Dict[int, float]:
        """
        Calculate cross-encoder relevance scores for a batch of documents.
        
        Args:
            query: Search query
            documents: List of (doc_idx, doc_dict) tuples to score
            
        Returns:
            Dictionary mapping doc_idx to normalized relevance score (0-0.3 range)
        """
        if not documents:
            return {}
        
        query_doc_pairs = [(query, doc['content']) for _, doc in documents]
        
        raw_scores = self.cross_encoder.predict(query_doc_pairs, batch_size=1)
        
        scores_array = np.array(raw_scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        score_range = max_score - min_score
        if score_range < 1e-6:
            normalized_scores = np.zeros_like(scores_array)
        else:
            normalized_scores = (scores_array - min_score) / score_range
        
        scaled_scores = normalized_scores * 0.05
        
        result = {}
        for i, (doc_idx, _) in enumerate(documents):
            result[doc_idx] = float(scaled_scores[i])
        
        return result
    
    def _calculate_recency_boost(self, date_str: str) -> float:
        """Calculate recency boost - ONLY for past dates"""
        if not date_str:
            return 0.0
        
        try:
            doc_date = parse_date_safe(date_str)
            now = datetime.now()
            
            # CRITICAL: Ignore future dates (appointments, typos, etc.)
            if doc_date > now:
                print(f"[WARNING] Future date detected: {date_str} - ignoring for recency boost")
                return 0.0
            
            days_ago = (now - doc_date).days
            recency_bonus = max(0, 0.03 - (days_ago / 365.0) * 0.03)
            return recency_bonus
        except:
            return 0.0
    
    def explain_rrf_score(self, doc_idx: int, query: str, rank_window: int = 150) -> Dict:
        """
        Explain how the RRF score was calculated for a specific document.
        
        Args:
            doc_idx: Document index
            query: Original query
            rank_window: Rank window used in RRF
            
        Returns:
            Dictionary with detailed score breakdown
        """
        semantic_ranking = self._get_semantic_ranking(query)
        keyword_ranking = self._get_keyword_ranking(query)
        
        explanation = {
            'document_id': self.documents[doc_idx].get('document_id', doc_idx),
            'rrf_k': self.k,
            'rank_window': rank_window,
            'rankings': {}
        }
        
        total_rrf = 0.0
        
        # Check semantic ranking
        semantic_rank = next((rank for rank, (idx, score) in enumerate(semantic_ranking, 1) 
                            if idx == doc_idx and rank <= rank_window), None)
        if semantic_rank:
            semantic_contribution = 1.0 / (self.k + semantic_rank)
            total_rrf += semantic_contribution
            explanation['rankings']['semantic'] = {
                'rank': semantic_rank,
                'original_score': next(score for idx, score in semantic_ranking if idx == doc_idx),
                'rrf_contribution': semantic_contribution
            }
        
        # Check keyword ranking
        keyword_rank = next((rank for rank, (idx, score) in enumerate(keyword_ranking, 1) 
                           if idx == doc_idx and rank <= rank_window), None)
        if keyword_rank:
            keyword_contribution = 1.0 / (self.k + keyword_rank)
            total_rrf += keyword_contribution
            explanation['rankings']['keyword'] = {
                'rank': keyword_rank,
                'original_score': next(score for idx, score in keyword_ranking if idx == doc_idx),
                'rrf_contribution': keyword_contribution
            }
        
        explanation['total_rrf_score'] = total_rrf
        return explanation

class RRFHybridRetriever:
    """
    Retriever adapter for the RRF hybrid search system.
    """
    
    def __init__(self, chroma_db, embedding_model, rrf_k: int = 60):
        """
        Initialize RRF hybrid retriever.
        
        Args:
            chroma_db: ChromaDB instance
            embedding_model: Sentence transformer model
            rrf_k: RRF smoothing constant
        """
        self.chroma_db = chroma_db
        self.rrf_search = RRFHybridSearch(embedding_model, k=rrf_k)
        self._initialize_from_chroma()
    
    def _initialize_from_chroma(self):
        """Initialize RRF search index from existing Chroma database."""
        
        print("[INFO] Initializing RRF hybrid search from Chroma...")
        
        # Get all documents from Chroma
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
            
            print(f"[SUCCESS] RRF hybrid search ready with {len(documents)} documents")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize RRF search: {e}")
            raise
    
    def retrieve_with_sources(self, query: str, max_references: int = 3, 
                            rank_window: int = 150,
                            enable_explanation: bool = False,
                            note_types: List[str] = None) -> Tuple[str, List[Dict]]:
        """
        Retrieve patient information using RRF hybrid search.
        
        Args:
            query: Search query
            max_references: Maximum number of results to return
            rank_window: RRF rank window size
            enable_explanation: Whether to include RRF score explanations
            
        Returns:
            Tuple of (formatted_content, sources_list)
        """
        
        # Perform RRF search
        results = self.rrf_search.rrf_search(
            query, 
            top_k=max_references * 2,  # Get more results for better selection
            rank_window=rank_window,
            note_types=note_types
        )
        
        # Convert to source format
        sources = []
        content_parts = ["# Patientoplysninger (RRF Hybrid Search)\n"]
        content_parts.append(f"*RRF sÃ¸gning fandt {len(results)} relevante kilder (k={self.rrf_search.k}).*\n")
        
        for i, result in enumerate(results[:max_references]):
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
                'cross_encoder_score': result['cross_encoder_score'],
                'recency_boost': result['recency_boost'],
                'rrf_k': result['rrf_k']
            }
            sources.append(source_ref)
            
            # Add to content display
            content_parts.append(f"## [{i+1}] {result['entry_type']} ({result['date']})")
            
            # Detailed RRF scoring information
            score_info = [
                f"**RRF: {result['rrf_score']:.4f}**",
                f"Total: {result['score']:.3f}",
                f"Sem: {result['semantic_score']:.3f} (#{result['semantic_rank'] or 'N/A'})",
                f"Key: {result['keyword_score']:.3f} (#{result['keyword_rank'] or 'N/A'})"
            ]
            
            if result['cross_encoder_score'] > 0:
                score_info.append(f"CE Relevance: +{result['cross_encoder_score']:.3f}")
            if result['recency_boost'] > 0:
                score_info.append(f"Recency: +{result['recency_boost']:.3f}")
            
            content_parts.append(f"*{' | '.join(score_info)} | Relevans: {relevance}%*\n")
            
            # Optional RRF explanation
            if enable_explanation and i < 2:  # Only explain top 2 results
                explanation = self.rrf_search.explain_rrf_score(
                    results[i]['chunk_index'], query, rank_window
                )
                content_parts.append("**RRF Score Breakdown:**")
                for ranking_type, details in explanation['rankings'].items():
                    content_parts.append(f"- {ranking_type.title()}: Rank #{details['rank']} --> "
                                       f"1/({self.rrf_search.k}+{details['rank']}) = {details['rrf_contribution']:.4f}")
                content_parts.append("")
            
            content_parts.append(result['content'])
            content_parts.append("\n---\n")
        
        full_content = "\n".join(content_parts)
        return full_content, sources