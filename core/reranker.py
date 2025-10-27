"""
Reranker model and operations for the Agentic RAG Medical Documentation System.
Handles cross-encoder reranker with robust error handling and GPU support.
Uses singleton pattern to prevent loading the same model multiple times.
"""

from sentence_transformers import CrossEncoder
from config.settings import RERANKER_MODEL_NAME, RERANKER_DEVICE
import torch

# Global cache for reranker model (singleton pattern)
_reranker_model_cache = None
_reranker_device_cache = None

def get_reranker() -> CrossEncoder:
    """
    Get the configured reranker model with robust error handling.
    Automatically handles GPU/CPU device selection with fallback.

    Uses torch.cuda.mem_get_info() to accurately detect free GPU memory,
    accounting for vLLM or other GPU usage. Automatically falls back to
    CPU if insufficient memory is available.

    Uses singleton pattern - only loads the model once and returns cached
    instance on subsequent calls.

    Returns:
        CrossEncoder: Configured cross-encoder reranker model instance.
    """
    global _reranker_model_cache, _reranker_device_cache

    # Return cached model if already loaded
    if _reranker_model_cache is not None:
        print(f"[INFO] Reusing cached reranker model (device: {_reranker_device_cache})")
        return _reranker_model_cache

    print(f"[INFO] Loading reranker model: {RERANKER_MODEL_NAME}")
    print(f"[INFO] Target device: {RERANKER_DEVICE}")

    # Determine device strategy
    device_attempts = []

    if RERANKER_DEVICE == "cuda" or RERANKER_DEVICE.startswith("cuda:"):
        if torch.cuda.is_available():
            # Try GPU first, then CPU as fallback
            device_attempts = [RERANKER_DEVICE, "cpu"]
        else:
            print(f"[WARNING] CUDA device requested but not available, using CPU")
            device_attempts = ["cpu"]
    else:
        device_attempts = [RERANKER_DEVICE]

    last_error = None

    for device in device_attempts:
        try:
            # Check GPU memory BEFORE attempting to load
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

            print(f"[INFO] Attempting to load reranker on {device}...")
            reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)

            # Verify the model works with a test prediction
            print(f"[INFO] Verifying reranker functionality with test prediction...")
            try:
                test_pairs = [['Test query', 'Test document']]
                test_scores = reranker.predict(test_pairs)
                print(f"[INFO] Test prediction successful (score: {test_scores[0]:.4f})")
            except Exception as test_error:
                error_msg = str(test_error).lower()
                print(f"[WARNING] Test prediction failed on {device}: {str(test_error)}")

                # Check if it's a CUDA/GPU error during usage
                if any(keyword in error_msg for keyword in ['cuda', 'nvml', 'gpu', 'device', 'caching']):
                    print(f"[WARNING] GPU error detected during model usage - model may be in inconsistent state")

                    # Clean up CUDA cache
                    if device.startswith("cuda") and torch.cuda.is_available():
                        print(f"[INFO] Clearing CUDA cache...")
                        torch.cuda.empty_cache()

                    # Don't cache this broken model
                    if device != "cpu":
                        print(f"[INFO] Will try CPU fallback...")
                        continue  # Move to next device
                    else:
                        raise Exception(f"Model failed even on CPU: {str(test_error)}")
                else:
                    # Non-GPU error during test
                    raise test_error

            print(f"[SUCCESS] Reranker loaded and verified successfully on {device}")

            # Cache the model for future use (singleton pattern)
            _reranker_model_cache = reranker
            _reranker_device_cache = device
            print(f"[INFO] Reranker model cached for reuse")

            return reranker

        except torch.cuda.OutOfMemoryError as e:
            print(f"[WARNING] GPU out of memory on {device}: {str(e)}")

            # Clean up CUDA cache before fallback
            if device.startswith("cuda") and torch.cuda.is_available():
                print(f"[INFO] Clearing CUDA cache...")
                torch.cuda.empty_cache()

            if device != "cpu":
                print(f"[INFO] Will try CPU fallback...")
                last_error = e
                continue  # Move to next device (CPU)
            else:
                raise Exception("Out of memory even on CPU - insufficient system resources")

        except Exception as e:
            error_msg = str(e).lower()
            print(f"[WARNING] Failed to load on {device}: {str(e)}")

            # Check if it's a GPU-related error
            if any(keyword in error_msg for keyword in ['cuda', 'gpu', 'memory', 'device', 'nvml', 'caching']):
                print(f"[INFO] GPU-related error detected, will try CPU fallback...")

                # Clean up CUDA cache before fallback
                if device.startswith("cuda") and torch.cuda.is_available():
                    print(f"[INFO] Clearing CUDA cache...")
                    torch.cuda.empty_cache()

                last_error = e
                continue  # Move to next device

            last_error = e

            if device == device_attempts[-1]:
                # All devices failed
                print(f"[ERROR] All reranker loading methods failed on all devices.")
                print(f"[INFO] This usually means:")
                print(f"  1. No internet connection for initial download")
                print(f"  2. Corrupted model cache")
                print(f"  3. HuggingFace Hub connectivity issues")
                print(f"  4. Insufficient GPU/CPU memory")
                raise Exception(f"Failed to load reranker after trying all strategies. Last error: {str(e)}")

    raise Exception(f"Failed to load reranker. Last error: {str(last_error)}")
