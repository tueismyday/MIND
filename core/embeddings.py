"""
Embedding model management for the MIND system.

This module handles HuggingFace embedding model loading with robust error
handling, automatic GPU/CPU device selection, and singleton pattern to prevent
loading the same model multiple times.

Architecture:
    The embedding model is loaded once and cached for reuse. The module
    automatically determines the optimal device (GPU or CPU) based on
    availability and memory constraints.

    Device Selection Strategy:
        1. Check preferred device (from config)
        2. If GPU: verify CUDA available and check memory
        3. If insufficient GPU memory: fallback to CPU
        4. Try multiple loading strategies with fallback

Key Functions:
    get_embedding_model: Main function to get cached or load embedding model

Dependencies:
    - langchain_huggingface: HuggingFace embeddings interface
    - core.device_manager: GPU/CPU device selection and memory management
    - config.settings: Model configuration

Singleton Pattern:
    The module uses global caching to ensure the embedding model is only
    loaded once. Subsequent calls to get_embedding_model() return the
    cached instance.

Example:
    >>> from core.embeddings import get_embedding_model
    >>> embeddings = get_embedding_model()
    >>> vector = embeddings.embed_query("medical text")
    >>> print(f"Embedding dimension: {len(vector)}")
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
import torch

from .device_manager import get_optimal_device, cleanup_cuda_cache
from .exceptions import ModelLoadingError
from .types import DeviceType
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE

# Configure logger for this module
logger = logging.getLogger(__name__)

# Global cache for embedding model (singleton pattern)
_cached_embedding_model: Optional[HuggingFaceEmbeddings] = None
_cached_embedding_device: Optional[str] = None

# Minimum free GPU memory required for embedding model (in GB)
# 0.6B parameter model ≈ 1.2GB + buffer
MIN_EMBEDDING_MEMORY_GB = 1.5


def _verify_model_functionality(
    embeddings: HuggingFaceEmbeddings,
    device: str
) -> bool:
    """
    Verify the embedding model works correctly with a test embedding.

    Args:
        embeddings: The embedding model to test
        device: Device the model is loaded on

    Returns:
        True if model works correctly, False otherwise

    Raises:
        ModelLoadingError: If model fails even on CPU (critical failure)
    """
    logger.debug("Verifying embedding model functionality")

    try:
        test_text = "Test embedding"
        test_vector = embeddings.embed_query(test_text)
        logger.info(f"Test embedding successful (dimension: {len(test_vector)})")
        return True

    except Exception as test_error:
        error_msg = str(test_error).lower()
        logger.warning(f"Test embedding failed on {device}: {str(test_error)}")

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
                    model_name=EMBEDDING_MODEL_NAME,
                    device="cpu"
                )
        else:
            # Non-GPU error during test - re-raise
            raise ModelLoadingError(
                f"Model verification failed: {str(test_error)}",
                model_name=EMBEDDING_MODEL_NAME,
                device=device
            )


def _attempt_model_load(
    device: str,
    attempt_number: int,
    total_attempts: int
) -> Optional[HuggingFaceEmbeddings]:
    """
    Attempt to load the embedding model with a specific strategy.

    Tries different loading strategies in order:
    1. Use system default cache (most reliable)
    2. Force re-download if cached version is corrupted
    3. Try with trust_remote_code for some models

    Args:
        device: Device to load model on
        attempt_number: Current attempt number (1-indexed)
        total_attempts: Total number of attempts to try

    Returns:
        HuggingFaceEmbeddings instance if successful, None otherwise
    """
    loading_strategies = [
        # 1. Use system default cache (most reliable)
        {"model_kwargs": {'device': device}},

        # 2. Force re-download if cached version is corrupted
        {"model_kwargs": {'device': device}, 'cache_folder': None},

        # 3. Try with trust_remote_code for some models
        {"model_kwargs": {'device': device, 'trust_remote_code': True}},
    ]

    if attempt_number > len(loading_strategies):
        return None

    kwargs = loading_strategies[attempt_number - 1]

    logger.info(
        f"Attempting to load embedding model on {device} "
        f"(strategy {attempt_number}/{total_attempts})"
    )

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            **kwargs
        )

        # Verify device placement
        if hasattr(embeddings, 'client') and hasattr(embeddings.client, 'device'):
            actual_device = str(embeddings.client.device)
            logger.debug(f"Model device verified: {actual_device}")

        return embeddings

    except Exception as e:
        error_msg = str(e).lower()
        logger.warning(f"Loading strategy {attempt_number} on {device} failed: {str(e)}")

        # Check if it's a GPU-related error
        if any(keyword in error_msg for keyword in ['cuda', 'gpu', 'memory', 'device', 'nvml', 'caching']):
            logger.info("GPU-related error detected during loading")

            # Clean up CUDA cache before giving up on this device
            if device.startswith("cuda"):
                logger.info("Clearing CUDA cache")
                cleanup_cuda_cache()

        return None


def _load_on_device(device: str) -> Optional[HuggingFaceEmbeddings]:
    """
    Load embedding model on a specific device with multiple strategies.

    Tries multiple loading strategies and verifies the model works before
    returning it.

    Args:
        device: Device to load model on ("cpu", "cuda", "cuda:0", etc.)

    Returns:
        HuggingFaceEmbeddings instance if successful, None if all strategies fail
    """
    logger.info(f"Attempting to load embedding model on {device}")

    total_attempts = 3

    for attempt in range(1, total_attempts + 1):
        embeddings = _attempt_model_load(device, attempt, total_attempts)

        if embeddings is not None:
            # Verify model works before declaring success
            if _verify_model_functionality(embeddings, device):
                logger.info(f"Embedding model loaded and verified successfully on {device}")
                return embeddings
            else:
                # Verification failed, try next strategy or device
                logger.warning(f"Model verification failed for strategy {attempt}")
                continue

    # All strategies failed for this device
    logger.warning(f"All loading strategies failed on {device}")
    return None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Get the configured embedding model with robust error handling.

    This function automatically handles GPU/CPU device selection with fallback,
    uses singleton pattern to cache the model, and implements multiple loading
    strategies for robustness.

    Device Selection:
        1. Uses device specified in config (EMBEDDING_DEVICE)
        2. Checks GPU memory availability before attempting GPU load
        3. Automatically falls back to CPU if GPU unavailable or insufficient memory
        4. Tries multiple loading strategies per device

    Singleton Pattern:
        Model is loaded once and cached. Subsequent calls return the cached
        instance immediately without reloading.

    Returns:
        HuggingFaceEmbeddings: Configured and verified embedding model instance

    Raises:
        ModelLoadingError: If model fails to load on all devices and strategies

    Example:
        >>> embeddings = get_embedding_model()
        >>> vector = embeddings.embed_query("diabetes mellitus")
        >>> print(f"Vector dimension: {len(vector)}")
    """
    global _cached_embedding_model, _cached_embedding_device

    # Return cached model if already loaded (singleton pattern)
    if _cached_embedding_model is not None:
        logger.info(
            f"Reusing cached embedding model (device: {_cached_embedding_device})"
        )
        return _cached_embedding_model

    logger.info("=" * 60)
    logger.info("Loading Embedding Model")
    logger.info("=" * 60)
    logger.info(f"Model: {EMBEDDING_MODEL_NAME}")
    logger.info(f"Target device: {EMBEDDING_DEVICE}")

    # Determine optimal device (handles GPU memory checking and fallback)
    try:
        optimal_device = get_optimal_device(
            preferred_device=EMBEDDING_DEVICE,
            min_memory_gb=MIN_EMBEDDING_MEMORY_GB,
            model_name="embedding",
            allow_cpu_fallback=True
        )
        logger.info(f"Selected device: {optimal_device}")
    except Exception as e:
        logger.error(f"Failed to determine optimal device: {e}")
        # Default to CPU if device selection fails
        optimal_device = "cpu"
        logger.info("Defaulting to CPU due to device selection error")

    # Try to load on the optimal device
    embeddings = _load_on_device(optimal_device)

    if embeddings is not None:
        # Success! Cache and return
        _cached_embedding_model = embeddings
        _cached_embedding_device = optimal_device
        logger.info("Embedding model cached for reuse")
        logger.info("=" * 60)
        return embeddings

    # If optimal device failed and it wasn't CPU, try CPU as last resort
    if optimal_device != "cpu":
        logger.warning(f"Failed to load on {optimal_device}, trying CPU as last resort")
        embeddings = _load_on_device("cpu")

        if embeddings is not None:
            _cached_embedding_model = embeddings
            _cached_embedding_device = "cpu"
            logger.info("Embedding model cached for reuse")
            logger.info("=" * 60)
            return embeddings

    # All devices and strategies failed
    logger.error("All embedding loading methods failed on all devices")
    logger.error("This usually indicates:")
    logger.error("  1. No internet connection for initial model download")
    logger.error("  2. Corrupted model cache")
    logger.error("  3. HuggingFace Hub connectivity issues")
    logger.error("  4. Insufficient GPU/CPU memory")
    logger.error("=" * 60)

    raise ModelLoadingError(
        "Failed to load embedding model after trying all strategies and devices",
        model_name=EMBEDDING_MODEL_NAME
    )
