"""
vLLM configuration for the MIND medical documentation system.

This module configures vLLM (vLarge Language Model) settings for both
server mode (external API) and local mode (in-Python instance).
"""

import logging
import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .exceptions import VLLMConfigurationError

logger = logging.getLogger(__name__)

# Type alias for vLLM modes
VLLMMode = Literal["server", "local"]


class VLLMConfig(BaseModel):
    """
    Configuration for vLLM language model operations.

    Supports two modes:
        - "server": Connect to external vLLM server via OpenAI-compatible API
        - "local": Load vLLM model directly in Python process

    Server Mode:
        Uses OpenAI-compatible API to connect to a running vLLM server.
        Requires vllm_server_url and vllm_model_name.
        Supports: temperature, top_p, presence_penalty

    Local Mode:
        Loads vLLM model directly in the Python process.
        Requires additional GPU configuration parameters.
        Supports: ALL sampling parameters including top_k and min_p

    Attributes:
        vllm_mode: Operating mode ("server" or "local")
        vllm_server_url: URL for server mode
        vllm_model_name: HuggingFace model identifier
        vllm_gpu_memory_utilization: GPU memory fraction (local mode)
        vllm_max_model_len: Maximum context length (local mode)
        vllm_max_num_seqs: Max parallel sequences (local mode)

    Environment Variables:
        MIND_VLLM_MODE: Override vLLM mode
        MIND_VLLM_SERVER_URL: Override server URL
        MIND_VLLM_MODEL_NAME: Override model name
        MIND_VLLM_GPU_MEMORY_UTILIZATION: Override GPU memory fraction
        MIND_VLLM_MAX_MODEL_LEN: Override max context length
        MIND_VLLM_MAX_NUM_SEQS: Override max sequences

    Example:
        >>> config = VLLMConfig()
        >>> print(config.vllm_mode)
        'server'
        >>> print(config.vllm_server_url)
        'http://localhost:8000'
    """

    vllm_mode: VLLMMode = Field(
        default="server",
        description="vLLM operating mode: 'server' or 'local'"
    )

    vllm_server_url: str = Field(
        default="http://localhost:8000",
        description="URL for vLLM server (server mode only)"
    )

    vllm_model_name: str = Field(
        default="cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
        description="HuggingFace model identifier"
    )

    # Local mode configuration
    vllm_gpu_memory_utilization: float = Field(
        default=0.90,
        ge=0.1,
        le=0.98,
        description="GPU memory fraction to use (local mode only)"
    )

    vllm_max_model_len: int = Field(
        default=14000,
        ge=512,
        le=128000,
        description="Maximum context length (local mode only)"
    )

    vllm_max_num_seqs: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Maximum parallel sequences (local mode only)"
    )

    model_config = {
        "validate_assignment": True,
        "use_enum_values": True
    }

    def __init__(self, **data):
        """
        Initialize VLLMConfig with environment variable support.

        Reads configuration from MIND_* prefixed environment variables
        and falls back to non-prefixed versions for compatibility.
        """
        # Check for environment variable overrides
        env_overrides = {}

        # vLLM mode
        if 'MIND_VLLM_MODE' in os.environ:
            env_overrides['vllm_mode'] = os.environ['MIND_VLLM_MODE'].lower()
        elif 'VLLM_MODE' in os.environ:
            env_overrides['vllm_mode'] = os.environ['VLLM_MODE'].lower()

        # Server URL
        if 'MIND_VLLM_SERVER_URL' in os.environ:
            env_overrides['vllm_server_url'] = os.environ['MIND_VLLM_SERVER_URL']
        elif 'VLLM_SERVER_URL' in os.environ:
            env_overrides['vllm_server_url'] = os.environ['VLLM_SERVER_URL']

        # Model name
        if 'MIND_VLLM_MODEL_NAME' in os.environ:
            env_overrides['vllm_model_name'] = os.environ['MIND_VLLM_MODEL_NAME']
        elif 'VLLM_MODEL_NAME' in os.environ:
            env_overrides['vllm_model_name'] = os.environ['VLLM_MODEL_NAME']

        # GPU memory utilization (local mode)
        if 'MIND_VLLM_GPU_MEMORY_UTILIZATION' in os.environ:
            env_overrides['vllm_gpu_memory_utilization'] = float(
                os.environ['MIND_VLLM_GPU_MEMORY_UTILIZATION']
            )
        elif 'VLLM_GPU_MEMORY_UTILIZATION' in os.environ:
            env_overrides['vllm_gpu_memory_utilization'] = float(
                os.environ['VLLM_GPU_MEMORY_UTILIZATION']
            )

        # Max model length (local mode)
        if 'MIND_VLLM_MAX_MODEL_LEN' in os.environ:
            env_overrides['vllm_max_model_len'] = int(
                os.environ['MIND_VLLM_MAX_MODEL_LEN']
            )
        elif 'VLLM_MAX_MODEL_LEN' in os.environ:
            env_overrides['vllm_max_model_len'] = int(
                os.environ['VLLM_MAX_MODEL_LEN']
            )

        # Max num seqs (local mode)
        if 'MIND_VLLM_MAX_NUM_SEQS' in os.environ:
            env_overrides['vllm_max_num_seqs'] = int(
                os.environ['MIND_VLLM_MAX_NUM_SEQS']
            )
        elif 'VLLM_MAX_NUM_SEQS' in os.environ:
            env_overrides['vllm_max_num_seqs'] = int(
                os.environ['VLLM_MAX_NUM_SEQS']
            )

        # Merge environment overrides with provided data
        merged_data = {**data, **env_overrides}

        super().__init__(**merged_data)

    @field_validator('vllm_mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate vLLM mode is valid."""
        valid_modes = ["server", "local"]
        if v not in valid_modes:
            raise VLLMConfigurationError(
                f"Invalid vLLM mode: {v}. Must be one of {valid_modes}"
            )
        return v

    @field_validator('vllm_server_url')
    @classmethod
    def validate_server_url(cls, v: str) -> str:
        """Validate server URL format."""
        if not v:
            raise VLLMConfigurationError("vLLM server URL cannot be empty")

        # Basic URL validation
        if not v.startswith(('http://', 'https://')):
            raise VLLMConfigurationError(
                f"vLLM server URL must start with http:// or https://, got: {v}"
            )

        return v.rstrip('/')  # Remove trailing slash for consistency

    @field_validator('vllm_model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise VLLMConfigurationError("vLLM model name cannot be empty")
        return v.strip()

    @field_validator('vllm_gpu_memory_utilization')
    @classmethod
    def validate_gpu_memory(cls, v: float) -> float:
        """Validate GPU memory utilization is reasonable."""
        if v < 0.1:
            raise VLLMConfigurationError(
                f"GPU memory utilization too low: {v}. Must be at least 0.1"
            )
        if v > 0.95:
            logger.warning(
                f"Very high GPU memory utilization ({v}). "
                "This may not leave enough memory for embeddings/reranker. "
                "Consider using 0.70-0.75 when running embeddings on GPU."
            )
        return v

    @field_validator('vllm_max_model_len')
    @classmethod
    def validate_max_len(cls, v: int) -> int:
        """Validate maximum model length is reasonable."""
        if v < 512:
            raise VLLMConfigurationError(
                f"Maximum model length too small: {v}. Must be at least 512"
            )
        if v > 128000:
            logger.warning(
                f"Very large maximum model length ({v}). "
                "This will require significant GPU memory."
            )
        return v

    def is_server_mode(self) -> bool:
        """Check if running in server mode."""
        return self.vllm_mode == "server"

    def is_local_mode(self) -> bool:
        """Check if running in local mode."""
        return self.vllm_mode == "local"

    def get_vllm_summary(self) -> str:
        """
        Get a summary of the vLLM configuration.

        Returns:
            Human-readable summary string
        """
        if self.is_server_mode():
            return f"vLLM Server Mode: {self.vllm_model_name} @ {self.vllm_server_url}"
        else:
            return (
                f"vLLM Local Mode: {self.vllm_model_name}, "
                f"GPU: {self.vllm_gpu_memory_utilization:.0%}, "
                f"max_len: {self.vllm_max_model_len}"
            )

    def log_configuration(self) -> None:
        """Log the current vLLM configuration."""
        logger.info("vLLM Configuration:")
        logger.info(f"  Mode: {self.vllm_mode}")
        logger.info(f"  Model: {self.vllm_model_name}")

        if self.is_server_mode():
            logger.info(f"  Server URL: {self.vllm_server_url}")
        else:
            logger.info(f"  GPU Memory Utilization: {self.vllm_gpu_memory_utilization}")
            logger.info(f"  Max Model Length: {self.vllm_max_model_len}")
            logger.info(f"  Max Num Seqs: {self.vllm_max_num_seqs}")
