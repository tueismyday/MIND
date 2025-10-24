"""
Enhanced patient information retrieval tools with RRF hybrid search and source references.
Handles patient record search with Reciprocal Rank Fusion combining semantic + keyword search.
"""

import os
from datetime import datetime
from sentence_transformers import util, SentenceTransformer
from langchain_core.tools import tool
from typing import List, Dict, Tuple, Optional

from core.database import db_manager
from config.settings import DANISH_CLINICAL_CATEGORIES, INITIAL_RETRIEVAL_K, FINAL_RETRIEVAL_K, SIMILARITY_SCORE_THRESHOLD, EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE
from utils.text_processing import parse_date_safe

# Import RRF hybrid search components
try:
    from tools.hybrid_search import RRFHybridRetriever
    RRF_SEARCH_AVAILABLE = True
    print("[INFO] RRF Hybrid Search available")
except ImportError:
    try:
        # Fallback to original lightweight hybrid search
        from tools.hybrid_search import RRFHybridRetriever
        RRF_SEARCH_AVAILABLE = False
        print("[INFO] Using fallback lightweight hybrid search")
    except ImportError:
        try:
            from rank_bm25 import BM25Okapi
            from sklearn.feature_extraction.text import TfidfVectorizer
            RRF_SEARCH_AVAILABLE = False
            print("[INFO] BM25 libraries available, but no hybrid search modules")
        except ImportError:
            RRF_SEARCH_AVAILABLE = False
            print("[WARNING] No hybrid search libraries available. Install with: pip install rank-bm25 scikit-learn")

class PatientInfoResult:
    """Container for patient information with source references and RRF details."""
    
    def __init__(self, content: str, sources: List[Dict], max_references: int = 3, 
                 search_method: str = "rrf", rrf_details: Dict = None):
        self.content = content
        self.sources = sources[:max_references]  # Limit to top X sources
        self.max_references = max_references
        self.search_method = search_method
        self.rrf_details = rrf_details or {}
    
    def get_formatted_content(self) -> str:
        """Get content with embedded source references and RRF details."""
        if not self.sources:
            return self.content
            
        # Add reference markers to content
        content_with_refs = self.content
        
        # Add reference list at the end
        ref_section = f"\n\n--- KILDER ({self.search_method.upper()}) ---"
        
        if self.search_method == "rrf" and self.rrf_details:
            ref_section += f"\nRRF Parameters: k={self.rrf_details.get('k', 60)}, "
            ref_section += f"window={self.rrf_details.get('rank_window', 100)}"
        
        for i, source in enumerate(self.sources, 1):
            timestamp = source.get('timestamp', 'Ukendt tidspunkt')
            entry_type = source.get('entry_type', 'Note')
            relevance = source.get('relevance', 0)
            
            ref_section += f"\n[{i}] {entry_type} - {timestamp} (Relevans: {relevance}%)"
            
            # Add RRF-specific details if available
            if self.search_method == "rrf" and 'rrf_score' in source:
                rrf_score = source['rrf_score']
                semantic_rank = source.get('semantic_rank', 'N/A')
                keyword_rank = source.get('keyword_rank', 'N/A')
                ref_section += f"\n    RRF: {rrf_score:.4f} (Semantic: #{semantic_rank}, Keyword: #{keyword_rank})"
            elif 'hybrid_score' in source:
                # Fallback to old hybrid score display
                ref_section += f" [Hybrid: {source['hybrid_score']:.3f}]"
            
            # Add snippet if available
            snippet = source.get('snippet', '')[:100]
            if snippet:
                ref_section += f"\n    Uddrag: \"{snippet}...\""
        
        return content_with_refs + ref_section
    
    def get_references_summary(self) -> str:
        """Get a summary of the references used."""
        if not self.sources:
            return "Ingen kilder fundet."
            
        summary = f"Baseret pÃ¥ {len(self.sources)} kilder fra patientjournalen ({self.search_method}):"
        for source in self.sources:
            timestamp = source.get('timestamp', 'Ukendt')
            entry_type = source.get('entry_type', 'Note')
            
            if self.search_method == "rrf" and 'rrf_score' in source:
                rrf_score = source['rrf_score']
                summary += f"\nâ€¢ {entry_type} ({timestamp}) [RRF: {rrf_score:.4f}]"
            else:
                summary += f"\nâ€¢ {entry_type} ({timestamp})"
        
        return summary
    
    def get_rrf_analysis(self) -> str:
        """Get detailed RRF analysis if available."""
        if self.search_method != "rrf" or not self.rrf_details:
            return "RRF analyse ikke tilgængelig."
        
        analysis = "ðŸ“Š **RRF (Reciprocal Rank Fusion) Analyse:**\n"
        analysis += f"- Smoothing konstant (k): {self.rrf_details.get('k', 60)}\n"
        analysis += f"- Rank vindue: {self.rrf_details.get('rank_window', 100)}\n"
        analysis += f"- Antal resultater fusioneret: {len(self.sources)}\n\n"
        
        analysis += "**Top resultater:**\n"
        for i, source in enumerate(self.sources[:3], 1):
            if 'rrf_score' in source:
                analysis += f"{i}. RRF Score: {source['rrf_score']:.4f}\n"
                if source.get('semantic_rank'):
                    analysis += f"   Semantisk rank: #{source['semantic_rank']}\n"
                if source.get('keyword_rank'):
                    analysis += f"   Keyword rank: #{source['keyword_rank']}\n"
        
        return analysis

class RRFPatientRetriever:
    """
    Enhanced retriever that prioritizes RRF hybrid search with intelligent fallbacks.
    """
    
    def __init__(self, rrf_k: int = 60):
        """
        Initialize RRF patient retriever.
        
        Args:
            rrf_k: RRF smoothing constant (default: 60)
        """
        self.rrf_retriever = None
        self.fallback_retriever = None
        self.use_rrf_search = False
        self.rrf_k = rrf_k
        
        # Load embedding model (needed for all search methods)
        print("[INFO] Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
        
        # Try to initialize RRF search first
        if RRF_SEARCH_AVAILABLE:
            try:
                self._initialize_rrf_search()
            except Exception as e:
                print(f"[ERROR] RRF initialization failed: {e}")
                print("[INFO] Attempting to initialize fallback search...")
                self._initialize_fallback_search()
    
    def _initialize_rrf_search(self):
        """Initialize RRF hybrid search."""
        from tools.hybrid_search import RRFHybridRetriever
        
        self.rrf_retriever = RRFHybridRetriever(
            db_manager.patient_db, 
            self.embedding_model,
            rrf_k=self.rrf_k
        )
        self.use_rrf_search = True
        print(f"[SUCCESS] RRF hybrid search initialized with k={self.rrf_k}")
    
    def _initialize_fallback_search(self):
        """Initialize fallback search (lightweight hybrid or Chroma)."""
        try:
            # Try lightweight hybrid search first
            from tools.hybrid_search import RRFHybridRetriever
            self.fallback_retriever = RRFHybridRetriever(
                db_manager.patient_db, 
                self.embedding_model
            )
            print("[INFO] Fallback: Lightweight hybrid search initialized")
        except Exception as e:
            print(f"[INFO] Fallback initialization failed: {e}")
            print("[INFO] Fallback: Using pure Chroma search")
    
    
    def retrieve_with_sources(self, query: str, initial_k: int = INITIAL_RETRIEVAL_K,
                            final_k: int = FINAL_RETRIEVAL_K,
                            max_references: int = 3,
                            rank_window: int = 100,
                            enable_rrf_explanation: bool = False,
                            note_types: List[str] = None) -> PatientInfoResult:
        """
        Retrieve patient information using best available search method.
        
        Args:
            query: Search query
            initial_k: Number of documents for initial retrieval (Chroma fallback)
            final_k: Number of documents after reranking (Chroma fallback)
            max_references: Maximum number of source references
            rank_window: RRF rank window size
            enable_rrf_explanation: Include detailed RRF score explanations
        """
        
        if self.use_rrf_search and self.rrf_retriever:
            return self._retrieve_with_rrf_search(query, max_references, rank_window, enable_rrf_explanation)
        elif self.fallback_retriever:
            return self._retrieve_with_hybrid_fallback(query, max_references)
        else:
            return self._retrieve_with_chroma(query, initial_k, final_k, max_references)
    
    def _retrieve_with_rrf_search(self, query: str, max_references: int, 
                                rank_window: int, enable_explanation: bool,
                                note_types: List[str] = None) -> PatientInfoResult:
        """Retrieve using RRF hybrid search."""
        
        print(f"[INFO] Using RRF hybrid search (k={self.rrf_k}, window={rank_window})...")
        
        try:
            content, sources = self.rrf_retriever.retrieve_with_sources(
                query, max_references, rank_window, enable_explanation
            )
            
            rrf_details = {
                'k': self.rrf_k,
                'rank_window': rank_window,
                'explanation_enabled': enable_explanation,
                'note_types_filter': note_types
            }
            
            return PatientInfoResult(content, sources, max_references, 
                                   search_method="rrf", rrf_details=rrf_details)
            
        except Exception as e:
            print(f"[ERROR] RRF search failed: {e}")
            print("[INFO] Falling back to alternative search")
            return self._retrieve_with_hybrid_fallback(query, max_references)
    
    def _retrieve_with_hybrid_fallback(self, query: str, max_references: int) -> PatientInfoResult:
        """Retrieve using lightweight hybrid search fallback."""
        
        if not self.fallback_retriever:
            return self._retrieve_with_chroma(query, INITIAL_RETRIEVAL_K, FINAL_RETRIEVAL_K, max_references)
        
        print("[INFO] Using lightweight hybrid search fallback...")
        
        try:
            content, sources = self.fallback_retriever.retrieve_with_sources(query, max_references)
            return PatientInfoResult(content, sources, max_references, search_method="hybrid_fallback")
            
        except Exception as e:
            print(f"[ERROR] Hybrid fallback failed: {e}")
            print("[INFO] Falling back to Chroma search")
            return self._retrieve_with_chroma(query, INITIAL_RETRIEVAL_K, FINAL_RETRIEVAL_K, max_references)
    
    def _retrieve_with_chroma(self, query: str, initial_k: int, final_k: int, max_references: int, note_types: List[str] = None) -> PatientInfoResult:
        """Final fallback to original Chroma-based search."""
        
        print(f"[INFO] Using Chroma search with clinical reranking (k={initial_k}->{final_k})...")
        
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

        metadata_filter = None
        if note_types:
            # ChromaDB filter format: {"entry_type": {"$in": ["type1", "type2"]}}
            metadata_filter = {"entry_type": {"$in": note_types}}
        
        # Initial semantic search
        results_with_scores = db_manager.patient_db.similarity_search_with_score(query, k=initial_k, filter=metadata_filter)
        
        # Filter by similarity threshold
        filtered_results = [doc for doc, score in results_with_scores if score >= SIMILARITY_SCORE_THRESHOLD]
        
        if not filtered_results:
            return PatientInfoResult("Ingen relevant patientinformation fundet.", [], search_method="chroma")
        
        # Cache embeddings for reranking
        doc_embeddings = {}
        for doc in filtered_results:
            content = doc.page_content.lower()
            doc_embeddings[id(doc)] = self.embedding_model.encode(content, convert_to_tensor=True)
        
        # Ranking function with clinical reranking
        def rank_score_with_metadata(doc):
            content = doc.page_content.lower()
            content_embedding = doc_embeddings[id(doc)]
            
            # Base similarity score
            score = util.cos_sim(query_embedding, content_embedding)[0][0].item()
            
            # Category matching bonus
            category = doc.metadata.get("category", "").lower()
            for cat, keywords in DANISH_CLINICAL_CATEGORIES.items():
                if cat in category or any(kw in content for kw in keywords):
                    score += 0.3
                    break
            
            # Recency bonus
            date_str = doc.metadata.get("date", "")
            try:
                doc_date = parse_date_safe(date_str)
                days_ago = (datetime.now() - doc_date).days
                recency_bonus = max(0, 0.3 - (days_ago / 365.0) * 0.3)
                score += recency_bonus
            except:
                pass
            return score
        
        # Rerank and limit results
        reranked_results = sorted(filtered_results, key=rank_score_with_metadata, reverse=True)
        limited_results = reranked_results[:final_k]
        
        # Build source references
        sources = []
        content_parts = ["# Patientoplysninger (Chroma + Clinical Reranking)\n"]
        content_parts.append(f"*Søgning fandt {len(filtered_results)} relevante kilder og viser de {len(limited_results)} mest relevante.*\n")
        
        scores = [rank_score_with_metadata(doc) for doc in limited_results]
        max_score = max(scores) if scores else 1
        min_score = min(scores) if scores else 0
        score_range = max(max_score - min_score, 0.001)
        
        for i, doc in enumerate(limited_results):
            raw_score = rank_score_with_metadata(doc)
            relevance = int((raw_score - min_score) / score_range * 100)
            
            # Extract metadata
            entry_type = doc.metadata.get("entry_type", "Note")
            date_str = doc.metadata.get("date", "") or "Ukendt dato"
            category = doc.metadata.get("category", "Ukategoriseret")
            content = doc.page_content.strip()
            
            # Add to sources list
            source_ref = {
                'timestamp': date_str,
                'entry_type': entry_type,
                'category': category,
                'relevance': relevance,
                'snippet': content[:150],
                'full_content': content,
                'chroma_score': raw_score
            }
            sources.append(source_ref)
            
            # Add to content
            content_parts.append(f"## [{i+1}] {entry_type} ({date_str})")
            
            # Metadata summary
            meta_summary = [f"**Relevans:** {relevance}% match"]
            
            # Time categorization
            try:
                doc_date = parse_date_safe(date_str)
                now = datetime.now()
                days_ago = (now - doc_date).days
                if days_ago < 7:
                    meta_summary.append("**Periode:** Meget nylig (<1 uge)")
                elif days_ago < 30:
                    meta_summary.append("**Periode:** Nylig (<1 måned)")
                elif days_ago < 180:
                    meta_summary.append("**Periode:** Inden for 6 måneder")
                else:
                    meta_summary.append(f"**Periode:** {days_ago // 365} Ã¥r gammel")
            except:
                meta_summary.append(f"**Dato:** {date_str}")
            
            meta_summary.append(f"**Kategori:** {category}")
            content_parts.append(f"*{' | '.join(meta_summary)}*\n")
            content_parts.append(content)
            content_parts.append("\n---\n")
        
        # Combine content
        full_content = "\n".join(content_parts)
        return PatientInfoResult(full_content, sources, max_references, search_method="chroma")

# Global retriever instance
_patient_retriever = None

def get_patient_retriever(rrf_k: int = 60) -> RRFPatientRetriever:
    """Get or create global patient retriever instance with RRF support."""
    global _patient_retriever
    if _patient_retriever is None:
        _patient_retriever = RRFPatientRetriever(rrf_k=rrf_k)
    return _patient_retriever

# Enhanced tool functions with RRF support
@tool
def retrieve_patient_info(query: str, initial_k: int = INITIAL_RETRIEVAL_K, 
                         final_k: int = FINAL_RETRIEVAL_K, 
                         max_references: int = 3,
                         rrf_k: int = 60,
                         rank_window: int = 100,
                         enable_rrf_explanation: bool = False,
                         note_types: List[str] = None) -> str:
    """
    Retrieve relevant patient information using RRF hybrid search with detailed source references.
    
    Args:
        query (str): Natural language query about the patient
        initial_k (int): Number of documents to retrieve initially (Chroma fallback)
        final_k (int): Number of documents after reranking (Chroma fallback)
        max_references (int): Maximum number of source references to include
        rrf_k (int): RRF smoothing constant (typically 60)
        rank_window (int): RRF rank window size (typically 100)
        enable_rrf_explanation (bool): Include detailed RRF score breakdowns
    
    Returns:
        str: Formatted patient information with source references and RRF details
    """
    
    retriever = get_patient_retriever(rrf_k)
    result = retriever.retrieve_with_sources(
        query, initial_k, final_k, max_references, 
        rank_window, enable_rrf_explanation, 
        note_types=note_types
    )
    return result.get_formatted_content()

@tool
def retrieve_patient_info_with_rrf_analysis(query: str, max_references: int = 3,
                                           rrf_k: int = 60, rank_window: int = 100) -> str:
    """
    Retrieve patient information with detailed RRF analysis for debugging/optimization.
    
    Args:
        query (str): Natural language query about the patient
        max_references (int): Maximum number of source references
        rrf_k (int): RRF smoothing constant
        rank_window (int): RRF rank window size
    
    Returns:
        str: Patient information with detailed RRF analysis
    """
    
    retriever = get_patient_retriever(rrf_k)
    result = retriever.retrieve_with_sources(
        query, max_references=max_references, 
        rank_window=rank_window, enable_rrf_explanation=True
    )
    
    content = result.get_formatted_content()
    analysis = result.get_rrf_analysis()
    
    return f"{content}\n\n{analysis}"

def get_patient_info_with_sources(query: str, initial_k: int = INITIAL_RETRIEVAL_K,
                                final_k: int = FINAL_RETRIEVAL_K,
                                max_references: int = 3,
                                rrf_k: int = 60,
                                rank_window: int = 100,
                                note_types: List[str] = None) -> PatientInfoResult:
    """
    Core function to retrieve patient information with RRF hybrid search and source tracking.
    
    Returns PatientInfoResult object containing content, source references, and RRF details.
    """
    
    retriever = get_patient_retriever(rrf_k)
    return retriever.retrieve_with_sources(
        query, initial_k, final_k, max_references, rank_window, note_types=note_types
    )