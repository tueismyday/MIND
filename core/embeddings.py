"""
Embedding models and operations for the Agentic RAG Medical Documentation System.
Handles HuggingFace embeddings with robust error handling and GPU support.
Uses singleton pattern to prevent loading the same model multiple times.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE
import torch
import os

# Global cache for embedding model (singleton pattern)
_embedding_model_cache = None
_embedding_device_cache = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get the configured embedding model with robust error handling.
    Automatically handles GPU/CPU device selection with fallback.

    Uses torch.cuda.mem_get_info() to accurately detect free GPU memory,
    accounting for vLLM or other GPU usage. Automatically falls back to
    CPU if insufficient memory is available.

    Uses singleton pattern - only loads the model once and returns cached
    instance on subsequent calls.

    Returns:
        HuggingFaceEmbeddings: Configured embedding model instance.
    """
    global _embedding_model_cache, _embedding_device_cache

    # Return cached model if already loaded
    if _embedding_model_cache is not None:
        print(f"[INFO] Reusing cached embedding model (device: {_embedding_device_cache})")
        return _embedding_model_cache

    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"[INFO] Target device: {EMBEDDING_DEVICE}")

    # Determine device strategy
    device_attempts = []

    if EMBEDDING_DEVICE == "cuda" or EMBEDDING_DEVICE.startswith("cuda:"):
        if torch.cuda.is_available():
            # Try GPU first, then CPU as fallback
            device_attempts = [EMBEDDING_DEVICE, "cpu"]
        else:
            print(f"[WARNING] CUDA device requested but not available, using CPU")
            device_attempts = ["cpu"]
    else:
        device_attempts = [EMBEDDING_DEVICE]

    # Try different device and loading strategies
    for device in device_attempts:
        # Check GPU memory BEFORE attempting to load, to avoid wasteful attempts
        if device.startswith("cuda"):
            gpu_idx = 0 if device == "cuda" else int(device.split(":")[1])

            # Use mem_get_info() for ACTUAL free memory (includes vLLM usage)
            free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(gpu_idx)
            free_mem = free_mem_bytes / (1024**3)
            total_mem = total_mem_bytes / (1024**3)
            used_mem = total_mem - free_mem

            print(f"[EMBEDDING GPU] Total: {total_mem:.2f}GB, Used: {used_mem:.2f}GB, Free: {free_mem:.2f}GB")

            # Need at least 1.5GB free for embedding model (0.6B params ≈ 1.2GB + buffer)
            MIN_FREE_FOR_EMBEDDING = 1.5

            if free_mem < MIN_FREE_FOR_EMBEDDING:
                print(f"[EMBEDDING GPU] Insufficient free memory ({free_mem:.2f}GB < {MIN_FREE_FOR_EMBEDDING}GB)")
                print(f"[EMBEDDING GPU] Falling back to CPU")
                continue  # Skip to CPU

        attempts = [
            # 1. Use system default cache (most reliable)
            {"model_kwargs": {'device': device}},

            # 2. Force re-download if cached version is corrupted
            {"model_kwargs": {'device': device}, 'cache_folder': None},

            # 3. Try with trust_remote_code for some models
            {"model_kwargs": {'device': device, 'trust_remote_code': True}},
        ]

        for i, kwargs in enumerate(attempts, 1):
            try:
                print(f"[INFO] Attempting to load on {device} (method {i}/{len(attempts)})...")

                embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    **kwargs
                )

                print(f"[SUCCESS] Embeddings loaded successfully on {device}")

                # Verify device placement
                if hasattr(embeddings, 'client') and hasattr(embeddings.client, 'device'):
                    actual_device = str(embeddings.client.device)
                    print(f"[INFO] Model device verified: {actual_device}")

                # Cache the model for future use (singleton pattern)
                _embedding_model_cache = embeddings
                _embedding_device_cache = device
                print(f"[INFO] Embedding model cached for reuse")

                return embeddings

            except torch.cuda.OutOfMemoryError as e:
                print(f"[WARNING] GPU out of memory on {device}: {str(e)}")
                if device != "cpu":
                    print(f"[INFO] Will try CPU fallback...")
                    break  # Move to next device (CPU)
                else:
                    raise Exception("Out of memory even on CPU - insufficient system resources")

            except Exception as e:
                error_msg = str(e).lower()
                print(f"[WARNING] Method {i} on {device} failed: {str(e)}")

                # Check if it's a GPU-related error
                if any(keyword in error_msg for keyword in ['cuda', 'gpu', 'memory', 'device']):
                    print(f"[INFO] GPU-related error detected, will try CPU fallback...")
                    break  # Move to next device

                if i == len(attempts):
                    # Last attempt for this device failed
                    if device != device_attempts[-1]:
                        print(f"[INFO] Moving to next device fallback...")
                    else:
                        # All devices and methods failed
                        print(f"[ERROR] All embedding loading methods failed on all devices.")
                        print(f"[INFO] This usually means:")
                        print(f"  1. No internet connection for initial download")
                        print(f"  2. Corrupted model cache")
                        print(f"  3. HuggingFace Hub connectivity issues")
                        print(f"  4. Insufficient GPU/CPU memory")
                        raise Exception(f"Failed to load embeddings after trying all strategies. Last error: {str(e)}")

    return None  # Should never reach here