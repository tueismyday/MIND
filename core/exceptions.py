"""
Custom exceptions for the MIND core module.

This module defines domain-specific exception classes for core infrastructure
components including database operations, model loading, and device management.

Exception Hierarchy:
    CoreError
    ├── DatabaseConnectionError
    ├── ModelLoadingError
    │   └── InsufficientGPUMemoryError
    └── DeviceNotAvailableError

Example:
    >>> from core.exceptions import DatabaseConnectionError
    >>> try:
    ...     db.connect()
    ... except DatabaseConnectionError as e:
    ...     logger.error(f"Failed to connect: {e}")
"""


class CoreError(Exception):
    """
    Base exception for core module errors.

    All core-related exceptions inherit from this base class,
    allowing for broad exception handling when needed.
    """
    pass


class DatabaseConnectionError(CoreError):
    """
    Raised when connection to a vector database fails.

    This can occur due to missing database directories, corrupted
    ChromaDB instances, or permission issues.

    Args:
        message: Description of the connection failure
        db_type: Type of database that failed (e.g., "patient", "guideline")
    """

    def __init__(self, message: str, db_type: str = None):
        self.db_type = db_type
        full_message = f"{message}"
        if db_type:
            full_message = f"[{db_type} DB] {message}"
        super().__init__(full_message)


class ModelLoadingError(CoreError):
    """
    Raised when loading a machine learning model fails.

    This can occur due to network issues, insufficient memory,
    corrupted model cache, or incompatible model configurations.

    Args:
        message: Description of the loading failure
        model_name: Name/identifier of the model that failed to load
        device: Device on which loading was attempted
    """

    def __init__(self, message: str, model_name: str = None, device: str = None):
        self.model_name = model_name
        self.device = device
        full_message = message
        if model_name:
            full_message = f"Model '{model_name}': {full_message}"
        if device:
            full_message = f"{full_message} (device: {device})"
        super().__init__(full_message)


class InsufficientGPUMemoryError(ModelLoadingError):
    """
    Raised when there is insufficient GPU memory to load a model.

    This exception is raised when GPU memory checks indicate that
    available memory is below the minimum threshold required for
    loading the model.

    Args:
        message: Description of the memory issue
        required_gb: Minimum GB of free memory required
        available_gb: Actual GB of free memory available
        device: GPU device identifier
    """

    def __init__(
        self,
        message: str = None,
        required_gb: float = None,
        available_gb: float = None,
        device: str = None
    ):
        self.required_gb = required_gb
        self.available_gb = available_gb

        if message is None:
            message = "Insufficient GPU memory"
            if required_gb is not None and available_gb is not None:
                message = f"Insufficient GPU memory: required {required_gb:.2f}GB, available {available_gb:.2f}GB"

        super().__init__(message, device=device)


class DeviceNotAvailableError(CoreError):
    """
    Raised when a requested compute device is not available.

    This can occur when CUDA is requested but not installed,
    or when a specific GPU index is requested but doesn't exist.

    Args:
        message: Description of the device availability issue
        requested_device: The device that was requested
    """

    def __init__(self, message: str, requested_device: str = None):
        self.requested_device = requested_device
        full_message = message
        if requested_device:
            full_message = f"Device '{requested_device}' not available: {message}"
        super().__init__(full_message)
