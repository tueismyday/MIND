"""
Model configuration for the MIND medical documentation system.

This module configures ML models including embeddings, rerankers,
and device selection modes. Device selection logic itself is in
core/device_manager.py to separate configuration from business logic.
"""

import logging
import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .exceptions import InvalidDeviceModeError, ModelConfigurationError

logger = logging.getLogger(__name__)

# Type alias for device modes
DeviceMode = Literal["auto", "cpu", "cuda"]


class ModelConfig(BaseModel):
    """
    Configuration for ML models used in the MIND system.

    Attributes:
        embedding_model_name: HuggingFace identifier for embedding model
        reranker_model_name: HuggingFace identifier for reranker model
        embedding_device_mode: Device selection mode for embeddings
        reranker_device_mode: Device selection mode for reranker
        batch_size_rerank: Batch size for reranker processing

    Device Modes:
        - "auto": Automatic device selection with GPU memory check
        - "cpu": Force CPU usage
        - "cuda": Force GPU usage (cuda:0)

    Environment Variables:
        MIND_EMBEDDING_MODEL_NAME: Override embedding model
        MIND_RERANKER_MODEL_NAME: Override reranker model
        MIND_EMBEDDING_DEVICE_MODE: Override embedding device mode
        MIND_RERANKER_DEVICE_MODE: Override reranker device mode
        EMBEDDING_DEVICE: Legacy env var for backward compatibility

    Example:
        >>> config = ModelConfig()
        >>> print(config.embedding_model_name)
        'Qwen/Qwen3-Embedding-0.6B'
        >>> print(config.embedding_device_mode)
        'cpu'
    """

    embedding_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="HuggingFace embedding model identifier"
    )

    reranker_model_name: str = Field(
        default="Qwen/Qwen3-Reranker-0.6B",
        description="HuggingFace reranker model identifier"
    )

    embedding_device_mode: DeviceMode = Field(
        default="cpu",
        description="Device selection mode for embedding model"
    )

    reranker_device_mode: DeviceMode = Field(
        default="cpu",
        description="Device selection mode for reranker model"
    )

    batch_size_rerank: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Batch size for cross-encoder reranking"
    )

    model_config = {
        "validate_assignment": True,
        "use_enum_values": True
    }

    def __init__(self, **data):
        """
        Initialize ModelConfig with environment variable support.

        Supports both new MIND_* prefixed variables and legacy variables
        for backward compatibility.
        """
        # Check for environment variable overrides
        env_overrides = {}

        # Embedding model name
        if 'MIND_EMBEDDING_MODEL_NAME' in os.environ:
            env_overrides['embedding_model_name'] = os.environ['MIND_EMBEDDING_MODEL_NAME']

        # Reranker model name
        if 'MIND_RERANKER_MODEL_NAME' in os.environ:
            env_overrides['reranker_model_name'] = os.environ['MIND_RERANKER_MODEL_NAME']

        # Embedding device mode
        if 'MIND_EMBEDDING_DEVICE_MODE' in os.environ:
            env_overrides['embedding_device_mode'] = (
                os.environ['MIND_EMBEDDING_DEVICE_MODE'].lower()
            )
        elif 'EMBEDDING_DEVICE_MODE' in os.environ:
            # Fallback to non-prefixed version
            env_overrides['embedding_device_mode'] = (
                os.environ['EMBEDDING_DEVICE_MODE'].lower()
            )

        # Reranker device mode
        if 'MIND_RERANKER_DEVICE_MODE' in os.environ:
            env_overrides['reranker_device_mode'] = (
                os.environ['MIND_RERANKER_DEVICE_MODE'].lower()
            )
        elif 'RERANKER_DEVICE_MODE' in os.environ:
            # Fallback to non-prefixed version
            env_overrides['reranker_device_mode'] = (
                os.environ['RERANKER_DEVICE_MODE'].lower()
            )

        # Legacy EMBEDDING_DEVICE variable (force CPU if set)
        if os.environ.get('EMBEDDING_DEVICE', '').lower() == 'cpu':
            logger.info(
                "Legacy EMBEDDING_DEVICE=cpu detected, forcing CPU mode "
                "for both embedding and reranker"
            )
            env_overrides['embedding_device_mode'] = 'cpu'
            env_overrides['reranker_device_mode'] = 'cpu'

        # Merge environment overrides with provided data
        merged_data = {**data, **env_overrides}

        super().__init__(**merged_data)

    @field_validator('embedding_device_mode', 'reranker_device_mode')
    @classmethod
    def validate_device_mode(cls, v: str) -> str:
        """Validate device mode is one of the allowed values."""
        allowed_modes = ["auto", "cpu", "cuda"]
        if v not in allowed_modes:
            raise InvalidDeviceModeError(
                f"Invalid device mode: {v}. Must be one of {allowed_modes}"
            )
        return v

    @field_validator('embedding_model_name', 'reranker_model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ModelConfigurationError("Model name cannot be empty")
        return v.strip()

    @field_validator('batch_size_rerank')
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is reasonable."""
        if v < 1:
            raise ModelConfigurationError(
                f"Batch size must be at least 1, got {v}"
            )
        if v > 32:
            logger.warning(
                f"Large batch size {v} may cause memory issues. "
                "Consider using a smaller value."
            )
        return v

    def get_model_summary(self) -> str:
        """
        Get a summary of the model configuration.

        Returns:
            Human-readable summary string
        """
        return (
            f"Embedding: {self.embedding_model_name} ({self.embedding_device_mode}), "
            f"Reranker: {self.reranker_model_name} ({self.reranker_device_mode})"
        )

    def log_configuration(self) -> None:
        """Log the current model configuration."""
        logger.info("Model Configuration:")
        logger.info(f"  Embedding Model: {self.embedding_model_name}")
        logger.info(f"  Embedding Device Mode: {self.embedding_device_mode}")
        logger.info(f"  Reranker Model: {self.reranker_model_name}")
        logger.info(f"  Reranker Device Mode: {self.reranker_device_mode}")
        logger.info(f"  Reranker Batch Size: {self.batch_size_rerank}")
