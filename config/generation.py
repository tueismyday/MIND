"""
Document generation configuration for the MIND medical documentation system.

This module configures parameters for document generation including
LLM sampling parameters, validation cycles, and agent settings.
"""

import logging

from pydantic import BaseModel, Field, field_validator

from .exceptions import InvalidConfigValueError

logger = logging.getLogger(__name__)


class GenerationConfig(BaseModel):
    """
    Configuration for document generation operations.

    Controls LLM sampling parameters, validation cycles, and agent behavior.
    These parameters apply to all model calls (retrieve, generate, critique).

    Sampling Parameters:
        - temperature: Controls randomness (0.0 = deterministic, 1.0+ = random)
        - top_p: Nucleus sampling - cumulative probability cutoff
        - top_k: Top-k sampling - limits to top k tokens (LOCAL MODE ONLY)
        - min_p: Minimum probability threshold (LOCAL MODE ONLY, 0 = disabled)
        - presence_penalty: Penalizes token repetition (0.0-2.0)

    Note on Parameter Compatibility:
        - Server mode (vLLM via OpenAI API): Supports temperature, top_p, presence_penalty
        - Local mode (vLLM in-Python): Supports ALL parameters including top_k and min_p

    Attributes:
        temperature: Sampling temperature for generation
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (local mode only)
        min_p: Minimum probability threshold (local mode only)
        presence_penalty: Token repetition penalty
        default_validation_cycles: Default validation/revision cycles per subsection
        max_validation_cycles: Maximum allowed validation cycles
        min_validation_cycles: Minimum validation cycles
        max_agent_iterations: Maximum iterations for agent
        agent_early_stopping_method: Early stopping method for agent
        memory_max_token_limit: Maximum token limit for memory
        chat_history_limit: Maximum chat history entries

    Example:
        >>> config = GenerationConfig()
        >>> print(config.temperature)
        0.2
        >>> config.temperature = 0.5
    """

    # LLM sampling parameters
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0=deterministic, 1.0+=random)"
    )

    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling - cumulative probability cutoff"
    )

    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Top-k sampling - limits to top k tokens (LOCAL MODE ONLY)"
    )

    min_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold (LOCAL MODE ONLY, 0=disabled)"
    )

    presence_penalty: float = Field(
        default=1.5,
        ge=0.0,
        le=2.0,
        description="Penalizes token repetition (0.0=no penalty, 2.0=max penalty)"
    )

    # Validation parameters
    default_validation_cycles: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Default number of validation/revision cycles per subsection"
    )

    max_validation_cycles: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum allowed validation cycles"
    )

    min_validation_cycles: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Minimum validation cycles (at least one critique)"
    )

    # Agent configuration
    max_agent_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum iterations for agent operations"
    )

    agent_early_stopping_method: str = Field(
        default="generate",
        description="Early stopping method for agent"
    )

    memory_max_token_limit: int = Field(
        default=12000,
        ge=1000,
        le=100000,
        description="Maximum token limit for agent memory"
    )

    chat_history_limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum chat history entries to retain"
    )

    model_config = {
        "validate_assignment": True
    }

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is in reasonable range."""
        if v < 0.0 or v > 2.0:
            raise InvalidConfigValueError(
                f"Temperature must be between 0.0 and 2.0, got {v}"
            )
        if v > 1.0:
            logger.warning(
                f"High temperature ({v}) may produce very random outputs"
            )
        return v

    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        """Validate top_p is in valid range."""
        if v < 0.0 or v > 1.0:
            raise InvalidConfigValueError(
                f"top_p must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator('presence_penalty')
    @classmethod
    def validate_presence_penalty(cls, v: float) -> float:
        """Validate presence_penalty is in valid range."""
        if v < 0.0 or v > 2.0:
            raise InvalidConfigValueError(
                f"presence_penalty must be between 0.0 and 2.0, got {v}"
            )
        if v > 1.8:
            logger.warning(
                f"High presence penalty ({v}) may produce unusual outputs"
            )
        return v

    @field_validator('max_validation_cycles')
    @classmethod
    def validate_max_cycles(cls, v: int, info) -> int:
        """Ensure max_validation_cycles >= min_validation_cycles."""
        if 'min_validation_cycles' in info.data:
            min_cycles = info.data['min_validation_cycles']
            if v < min_cycles:
                raise InvalidConfigValueError(
                    f"max_validation_cycles ({v}) cannot be less than "
                    f"min_validation_cycles ({min_cycles})"
                )
        return v

    @field_validator('default_validation_cycles')
    @classmethod
    def validate_default_cycles(cls, v: int, info) -> int:
        """Ensure default is within min and max range."""
        data = info.data
        if 'min_validation_cycles' in data and v < data['min_validation_cycles']:
            raise InvalidConfigValueError(
                f"default_validation_cycles ({v}) cannot be less than "
                f"min_validation_cycles ({data['min_validation_cycles']})"
            )
        if 'max_validation_cycles' in data and v > data['max_validation_cycles']:
            raise InvalidConfigValueError(
                f"default_validation_cycles ({v}) cannot be greater than "
                f"max_validation_cycles ({data['max_validation_cycles']})"
            )
        return v

    def get_sampling_params_dict(self, local_mode: bool = False) -> dict:
        """
        Get sampling parameters as a dictionary.

        Args:
            local_mode: If True, include all parameters (local vLLM).
                       If False, exclude top_k and min_p (server mode).

        Returns:
            Dictionary of sampling parameters
        """
        params = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'presence_penalty': self.presence_penalty,
        }

        if local_mode:
            params['top_k'] = self.top_k
            params['min_p'] = self.min_p

        return params

    def get_generation_summary(self) -> str:
        """
        Get a summary of the generation configuration.

        Returns:
            Human-readable summary string
        """
        return (
            f"Generation: temp={self.temperature}, top_p={self.top_p}, "
            f"validation_cycles={self.default_validation_cycles}"
        )

    def log_configuration(self, local_mode: bool = False) -> None:
        """
        Log the current generation configuration.

        Args:
            local_mode: Whether running in local vLLM mode
        """
        logger.info("Generation Configuration:")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  Top P: {self.top_p}")
        if local_mode:
            logger.info(f"  Top K: {self.top_k} (local mode)")
            logger.info(f"  Min P: {self.min_p} (local mode)")
        logger.info(f"  Presence Penalty: {self.presence_penalty}")
        logger.info(f"  Default Validation Cycles: {self.default_validation_cycles}")
        logger.info(f"  Max Validation Cycles: {self.max_validation_cycles}")
        logger.info(f"  Min Validation Cycles: {self.min_validation_cycles}")
        logger.info(f"  Max Agent Iterations: {self.max_agent_iterations}")
        logger.info(f"  Memory Token Limit: {self.memory_max_token_limit}")
