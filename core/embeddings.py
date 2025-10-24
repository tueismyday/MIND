"""
Embedding models and operations for the Agentic RAG Medical Documentation System.
Handles HuggingFace embeddings with robust error handling.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE
import os

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get the configured embedding model with robust error handling.
    
    Returns:
        HuggingFaceEmbeddings: Configured embedding model instance.
    """
    
    # Try multiple approaches in order of preference
    attempts = [
        # 1. Use system default cache (most reliable)
        {"model_kwargs": {'device': EMBEDDING_DEVICE}},
        
        # 2. Force re-download if cached version is corrupted
        {"model_kwargs": {'device': EMBEDDING_DEVICE}, 'cache_folder': None},
        
        # 3. Try with trust_remote_code for some models
        {"model_kwargs": {'device': EMBEDDING_DEVICE, 'trust_remote_code': True}},
    ]
    
    for i, kwargs in enumerate(attempts, 1):
        try:
            print(f"[INFO] Attempting to load embeddings (method {i}/3)...")
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                **kwargs
            )
            print(f"[SUCCESS] Embeddings loaded successfully with method {i}")
            return embeddings
            
        except Exception as e:
            print(f"[WARNING] Method {i} failed: {str(e)}")
            if i == len(attempts):
                # Last attempt failed, provide helpful error message
                print(f"[ERROR] All embedding loading methods failed.")
                print(f"[INFO] This usually means:")
                print(f"  1. No internet connection for initial download")
                print(f"  2. Corrupted model cache")
                print(f"  3. HuggingFace Hub connectivity issues")
                raise Exception(f"Failed to load embeddings after {len(attempts)} attempts. Last error: {str(e)}")
    
    return None  # Should never reach here