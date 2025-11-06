"""
Cross-encoder reranker model management for the MIND system.

This module handles cross-encoder reranker model loading with robust error
handling, automatic GPU/CPU device selection, and singleton pattern to prevent
loading the same model multiple times.

Architecture:
    The reranker model is loaded once and cached for reuse. The module
    automatically determines the optimal device (GPU or CPU) based on
    availability and memory constraints, similar to the embeddings module.

    The reranker is used in the second stage of hybrid search to provide
    accurate relevance scoring between query-document pairs.

Key Functions:
    get_reranker_model: Main function to get cached or load reranker model

Dependencies:
    - sentence_transformers: Cross-encoder implementation
    - core.device_manager: GPU/CPU device selection and memory management
    - config.settings: Model configuration

Singleton Pattern:
    The module uses global caching to ensure the reranker model is only
    loaded once. Subsequent calls to get_reranker_model() return the
    cached instance.

Example:
    >>> from core.reranker import get_reranker_model
    >>> reranker = get_reranker_model()
    >>> scores = reranker.predict([
    ...     ['query text', 'document 1'],
    ...     ['query text', 'document 2']
    ... ])
    >>> print(f"Relevance scores: {scores}")
"""

import logging
from typing import List, Tuple, Optional

from sentence_transformers import CrossEncoder
import torch

from .device_manager import get_optimal_device, cleanup_cuda_cache
from .exceptions import ModelLoadingError
from .types import DeviceType
from config.settings import RERANKER_MODEL_NAME, RERANKER_DEVICE

# Configure logger for this module
logger = logging.getLogger(__name__)

# Global cache for reranker model (singleton pattern)
_cached_reranker_model: Optional[CrossEncoder] = None
_cached_reranker_device: Optional[str] = None

# Minimum free GPU memory required for reranker model (in GB)
# 0.6B parameter model ≈ 1.2GB + buffer
MIN_RERANKER_MEMORY_GB = 1.5


def _verify_reranker_functionality(
    reranker: CrossEncoder,
    device: str
) -> bool:
    """
    Verify the reranker model works correctly with a test prediction.

    Args:
        reranker: The reranker model to test
        device: Device the model is loaded on

    Returns:
        True if model works correctly, False otherwise

    Raises:
        ModelLoadingError: If model fails even on CPU (critical failure)
    """
    logger.debug("Verifying reranker model functionality")

    try:
        test_pairs = [['Test query', 'Test document']]
        test_scores = reranker.predict(test_pairs)
        logger.info(f"Test prediction successful (score: {test_scores[0]:.4f})")
        return True

    except Exception as test_error:
        error_msg = str(test_error).lower()
        logger.warning(f"Test prediction failed on {device}: {str(test_error)}")

        # Check if it's a CUDA/GPU error during usage
        if any(keyword in error_msg for keyword in ['cuda', 'nvml', 'gpu', 'device', 'caching']):
            logger.warning("GPU error detected during model usage - model may be in inconsistent state")

            # Clean up CUDA cache
            if device.startswith("cuda"):
                logger.info("Clearing CUDA cache after GPU error")
                cleanup_cuda_cache()

            # If not on CPU, we can try CPU fallback
            if device != "cpu":
                logger.info("Will try CPU fallback")
                return False
            else:
                # Model failed even on CPU - this is critical
                raise ModelLoadingError(
                    f"Model failed even on CPU: {str(test_error)}",
                    model_name=RERANKER_MODEL_NAME,
                    device="cpu"
                )
        else:
            # Non-GPU error during test - re-raise
            raise ModelLoadingError(
                f"Model verification failed: {str(test_error)}",
                model_name=RERANKER_MODEL_NAME,
                device=device
            )


def _load_reranker_on_device(device: str) -> Optional[CrossEncoder]:
    """
    Load reranker model on a specific device.

    Args:
        device: Device to load model on ("cpu", "cuda", "cuda:0", etc.)

    Returns:
        CrossEncoder instance if successful, None if loading fails
    """
    logger.info(f"Attempting to load reranker model on {device}")

    try:
        reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)
        logger.debug(f"Reranker model instantiated on {device}")

        # Verify model works before declaring success
        if _verify_reranker_functionality(reranker, device):
            logger.info(f"Reranker model loaded and verified successfully on {device}")
            return reranker
        else:
            # Verification failed
            logger.warning("Reranker model verification failed")
            return None

    except torch.cuda.OutOfMemoryError as e:
        logger.warning(f"GPU out of memory on {device}: {str(e)}")

        # Clean up CUDA cache
        if device.startswith("cuda"):
            logger.info("Clearing CUDA cache after OOM error")
            cleanup_cuda_cache()

        return None

    except Exception as e:
        error_msg = str(e).lower()
        logger.warning(f"Failed to load reranker on {device}: {str(e)}")

        # Check if it's a GPU-related error
        if any(keyword in error_msg for keyword in ['cuda', 'gpu', 'memory', 'device', 'nvml', 'caching']):
            logger.info("GPU-related error detected during loading")

            # Clean up CUDA cache
            if device.startswith("cuda"):
                logger.info("Clearing CUDA cache")
                cleanup_cuda_cache()

        return None


def get_reranker_model() -> CrossEncoder:
    """
    Get the configured cross-encoder reranker model with robust error handling.

    This function automatically handles GPU/CPU device selection with fallback,
    uses singleton pattern to cache the model, and verifies the model works
    before returning it.

    Device Selection:
        1. Uses device specified in config (RERANKER_DEVICE)
        2. Checks GPU memory availability before attempting GPU load
        3. Automatically falls back to CPU if GPU unavailable or insufficient memory

    Singleton Pattern:
        Model is loaded once and cached. Subsequent calls return the cached
        instance immediately without reloading.

    Returns:
        CrossEncoder: Configured and verified reranker model instance

    Raises:
        ModelLoadingError: If model fails to load on all devices

    Example:
        >>> reranker = get_reranker_model()
        >>> query_doc_pairs = [
        ...     ['diabetes treatment', 'insulin therapy protocol'],
        ...     ['diabetes treatment', 'cardiac surgery guidelines']
        ... ]
        >>> scores = reranker.predict(query_doc_pairs)
        >>> print(f"Relevance scores: {scores}")
    """
    global _cached_reranker_model, _cached_reranker_device

    # Return cached model if already loaded (singleton pattern)
    if _cached_reranker_model is not None:
        logger.info(
            f"Reusing cached reranker model (device: {_cached_reranker_device})"
        )
        return _cached_reranker_model

    logger.info("=" * 60)
    logger.info("Loading Reranker Model")
    logger.info("=" * 60)
    logger.info(f"Model: {RERANKER_MODEL_NAME}")
    logger.info(f"Target device: {RERANKER_DEVICE}")

    # Determine optimal device (handles GPU memory checking and fallback)
    try:
        optimal_device = get_optimal_device(
            preferred_device=RERANKER_DEVICE,
            min_memory_gb=MIN_RERANKER_MEMORY_GB,
            model_name="reranker",
            allow_cpu_fallback=True
        )
        logger.info(f"Selected device: {optimal_device}")
    except Exception as e:
        logger.error(f"Failed to determine optimal device: {e}")
        # Default to CPU if device selection fails
        optimal_device = "cpu"
        logger.info("Defaulting to CPU due to device selection error")

    # Try to load on the optimal device
    reranker = _load_reranker_on_device(optimal_device)

    if reranker is not None:
        # Success! Cache and return
        _cached_reranker_model = reranker
        _cached_reranker_device = optimal_device
        logger.info("Reranker model cached for reuse")
        logger.info("=" * 60)
        return reranker

    # If optimal device failed and it wasn't CPU, try CPU as last resort
    if optimal_device != "cpu":
        logger.warning(f"Failed to load on {optimal_device}, trying CPU as last resort")
        reranker = _load_reranker_on_device("cpu")

        if reranker is not None:
            _cached_reranker_model = reranker
            _cached_reranker_device = "cpu"
            logger.info("Reranker model cached for reuse")
            logger.info("=" * 60)
            return reranker

    # All devices failed
    logger.error("All reranker loading methods failed on all devices")
    logger.error("This usually indicates:")
    logger.error("  1. No internet connection for initial model download")
    logger.error("  2. Corrupted model cache")
    logger.error("  3. HuggingFace Hub connectivity issues")
    logger.error("  4. Insufficient GPU/CPU memory")
    logger.error("=" * 60)

    raise ModelLoadingError(
        "Failed to load reranker model after trying all strategies and devices",
        model_name=RERANKER_MODEL_NAME
    )
