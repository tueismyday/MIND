"""
Device management for GPU/CPU selection and memory monitoring.

This module provides centralized device selection and GPU memory management
for all models in the MIND system. It consolidates GPU availability checking,
memory monitoring, and device fallback logic.

Key Functions:
    check_gpu_memory: Check if GPU has sufficient free memory
    get_optimal_device: Determine optimal device for model loading
    cleanup_cuda_cache: Clear CUDA cache to free memory

Dependencies:
    - torch: For CUDA operations and memory management
    - logging: For structured logging of device operations

Example:
    >>> from core.device_manager import get_optimal_device
    >>> device = get_optimal_device(
    ...     preferred_device="cuda",
    ...     min_memory_gb=1.5,
    ...     model_name="embedding"
    ... )
    >>> print(f"Selected device: {device}")
"""

import logging
from typing import Dict, Tuple, Optional, Literal
import torch

from .exceptions import DeviceNotAvailableError, InsufficientGPUMemoryError

# Configure logger for this module
logger = logging.getLogger(__name__)

# Type alias for device types
DeviceType = Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]

# Default memory requirements (in GB)
DEFAULT_MIN_FREE_MEMORY_GB = 1.5


def check_gpu_memory(
    device_id: int = 0,
    min_required_gb: float = DEFAULT_MIN_FREE_MEMORY_GB
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if GPU has sufficient free memory.

    Uses torch.cuda.mem_get_info() to get accurate free memory information,
    accounting for all GPU usage including vLLM or other processes.

    Args:
        device_id: GPU device index (default: 0)
        min_required_gb: Minimum free memory required in gigabytes

    Returns:
        Tuple of (is_sufficient, memory_info_dict) where:
            - is_sufficient: Boolean indicating if memory is sufficient
            - memory_info_dict: Dictionary with 'total_gb', 'used_gb', 'free_gb' keys

    Raises:
        DeviceNotAvailableError: If CUDA is not available or device_id is invalid

    Example:
        >>> is_ok, info = check_gpu_memory(device_id=0, min_required_gb=2.0)
        >>> if is_ok:
        ...     print(f"GPU has {info['free_gb']:.2f}GB free")
    """
    if not torch.cuda.is_available():
        raise DeviceNotAvailableError(
            "CUDA is not available on this system",
            requested_device=f"cuda:{device_id}"
        )

    if device_id >= torch.cuda.device_count():
        raise DeviceNotAvailableError(
            f"GPU device index {device_id} not found. Available devices: {torch.cuda.device_count()}",
            requested_device=f"cuda:{device_id}"
        )

    try:
        # Get actual free memory (accounts for all processes)
        free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(device_id)
        free_gb = free_mem_bytes / (1024**3)
        total_gb = total_mem_bytes / (1024**3)
        used_gb = total_gb - free_gb

        memory_info = {
            'total_gb': total_gb,
            'used_gb': used_gb,
            'free_gb': free_gb
        }

        is_sufficient = free_gb >= min_required_gb

        logger.debug(
            "GPU memory check completed",
            extra={
                'device_id': device_id,
                'total_gb': f"{total_gb:.2f}",
                'used_gb': f"{used_gb:.2f}",
                'free_gb': f"{free_gb:.2f}",
                'min_required_gb': min_required_gb,
                'is_sufficient': is_sufficient
            }
        )

        return is_sufficient, memory_info

    except Exception as e:
        logger.error(f"Failed to check GPU memory for device {device_id}: {e}")
        raise DeviceNotAvailableError(
            f"Failed to query GPU memory: {str(e)}",
            requested_device=f"cuda:{device_id}"
        )


def get_optimal_device(
    preferred_device: str,
    min_memory_gb: float = DEFAULT_MIN_FREE_MEMORY_GB,
    model_name: str = "model",
    allow_cpu_fallback: bool = True
) -> str:
    """
    Determine the optimal device for model loading with automatic fallback.

    Attempts to use the preferred device (typically GPU) and automatically
    falls back to CPU if the GPU doesn't have sufficient memory or isn't
    available.

    Args:
        preferred_device: Preferred device ("cpu", "cuda", "cuda:0", etc.)
        min_memory_gb: Minimum free GPU memory required in GB
        model_name: Name of the model (for logging purposes)
        allow_cpu_fallback: Whether to fallback to CPU if GPU unavailable

    Returns:
        Device string to use for model loading

    Raises:
        DeviceNotAvailableError: If no suitable device is found
        InsufficientGPUMemoryError: If GPU requested but insufficient memory
            and CPU fallback is disabled

    Example:
        >>> device = get_optimal_device(
        ...     preferred_device="cuda",
        ...     min_memory_gb=1.5,
        ...     model_name="embedding",
        ...     allow_cpu_fallback=True
        ... )
    """
    logger.info(
        f"Determining optimal device for {model_name}",
        extra={'preferred_device': preferred_device, 'min_memory_gb': min_memory_gb}
    )

    # If CPU is preferred, return immediately
    if preferred_device == "cpu":
        logger.info(f"Using CPU as preferred device for {model_name}")
        return "cpu"

    # Handle CUDA devices
    if preferred_device == "cuda" or preferred_device.startswith("cuda:"):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning(
                f"CUDA requested for {model_name} but not available"
            )
            if allow_cpu_fallback:
                logger.info(f"Falling back to CPU for {model_name}")
                return "cpu"
            else:
                raise DeviceNotAvailableError(
                    "CUDA is not available",
                    requested_device=preferred_device
                )

        # Extract GPU device index
        device_id = 0 if preferred_device == "cuda" else int(preferred_device.split(":")[1])

        # Check GPU memory
        try:
            is_sufficient, memory_info = check_gpu_memory(device_id, min_memory_gb)

            logger.info(
                f"GPU memory status for {model_name}",
                extra={
                    'device': f"cuda:{device_id}",
                    'total_gb': f"{memory_info['total_gb']:.2f}",
                    'used_gb': f"{memory_info['used_gb']:.2f}",
                    'free_gb': f"{memory_info['free_gb']:.2f}",
                    'required_gb': min_memory_gb
                }
            )

            if is_sufficient:
                logger.info(f"Using {preferred_device} for {model_name}")
                return preferred_device
            else:
                logger.warning(
                    f"Insufficient GPU memory for {model_name}: "
                    f"{memory_info['free_gb']:.2f}GB < {min_memory_gb}GB"
                )
                if allow_cpu_fallback:
                    logger.info(f"Falling back to CPU for {model_name}")
                    return "cpu"
                else:
                    raise InsufficientGPUMemoryError(
                        required_gb=min_memory_gb,
                        available_gb=memory_info['free_gb'],
                        device=preferred_device
                    )

        except DeviceNotAvailableError as e:
            logger.error(f"Device check failed for {model_name}: {e}")
            if allow_cpu_fallback:
                logger.info(f"Falling back to CPU for {model_name}")
                return "cpu"
            else:
                raise

    # Unknown device type
    logger.warning(f"Unknown device type '{preferred_device}', defaulting to CPU")
    return "cpu"


def cleanup_cuda_cache() -> None:
    """
    Clear CUDA cache to free up GPU memory.

    This should be called after model loading failures or before attempting
    to load another model to maximize available GPU memory.

    Example:
        >>> from core.device_manager import cleanup_cuda_cache
        >>> try:
        ...     model = load_large_model()
        ... except OutOfMemoryError:
        ...     cleanup_cuda_cache()
        ...     model = load_smaller_model()
    """
    if torch.cuda.is_available():
        logger.debug("Clearing CUDA cache to free GPU memory")
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared")
    else:
        logger.debug("CUDA not available, skipping cache cleanup")


def log_device_info() -> None:
    """
    Log detailed information about available compute devices.

    Useful for debugging and understanding the compute environment
    at system startup.

    Example:
        >>> from core.device_manager import log_device_info
        >>> log_device_info()
    """
    logger.info("=" * 50)
    logger.info("Compute Device Information")
    logger.info("=" * 50)

    # CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {device_count}")

        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            logger.info(f"Device {i}: {device_name}")

            try:
                is_sufficient, memory_info = check_gpu_memory(i, 0)
                logger.info(
                    f"  Memory - Total: {memory_info['total_gb']:.2f}GB, "
                    f"Used: {memory_info['used_gb']:.2f}GB, "
                    f"Free: {memory_info['free_gb']:.2f}GB"
                )
            except Exception as e:
                logger.warning(f"  Could not query memory for device {i}: {e}")
    else:
        logger.info("No CUDA devices available, will use CPU")

    logger.info("=" * 50)
