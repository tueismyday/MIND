"""
Main configuration settings for the MIND medical documentation system.

This module provides the central MINDSettings class that aggregates all
configuration subsystems (paths, models, retrieval, generation, vLLM).

Configuration can be loaded from:
    - Environment variables (MIND_* prefix)
    - .env file
    - Direct instantiation with overrides

Example:
    >>> from config import get_settings
    >>> settings = get_settings()
    >>> print(settings.paths.data_dir)
    >>> print(settings.models.embedding_model_name)
    >>> print(settings.generation.temperature)
"""

import logging
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError
from .generation import GenerationConfig
from .models import ModelConfig
from .paths import PathConfig
from .retrieval import RetrievalConfig
from .vllm import VLLMConfig

logger = logging.getLogger(__name__)


class MINDSettings(BaseSettings):
    """
    Main configuration class for the MIND system.

    Aggregates all configuration subsystems into a single validated
    configuration object. Supports environment variable overrides
    and .env file loading.

    Attributes:
        paths: Path configuration (directories, files)
        models: Model configuration (embeddings, reranker, devices)
        retrieval: Retrieval configuration (RAG parameters)
        generation: Generation configuration (LLM sampling, validation)
        vllm: vLLM configuration (server/local mode)

    Environment Variables:
        All configuration can be overridden via environment variables
        with the MIND_ prefix. See individual config modules for details.

    Configuration File:
        Place a .env file in the project root to set default values.
        Environment variables take precedence over .env file.

    Example:
        >>> settings = MINDSettings()
        >>> settings.paths.ensure_directories()
        >>> print(settings.models.get_model_summary())
        >>> print(settings.vllm.get_vllm_summary())
    """

    # Configuration subsystems
    paths: PathConfig = Field(default_factory=PathConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MIND_",
        env_nested_delimiter="__",
        case_sensitive=False,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="ignore"
    )

    def validate_all(self) -> bool:
        """
        Validate all configuration subsystems.

        Returns:
            True if all validations pass

        Raises:
            ConfigurationError: If any validation fails
        """
        try:
            # Validate paths
            self.paths.validate_paths()

            # Validate models
            if not self.models.embedding_model_name:
                raise ConfigurationError("Embedding model name is required")
            if not self.models.reranker_model_name:
                raise ConfigurationError("Reranker model name is required")

            # Validate vLLM
            if self.vllm.is_server_mode():
                if not self.vllm.vllm_server_url:
                    raise ConfigurationError("vLLM server URL is required in server mode")

            logger.info("Configuration validation successful")
            return True

        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e

    def ensure_initialized(self) -> None:
        """
        Ensure the system is properly initialized.

        Creates necessary directories and validates configuration.
        Call this before using the MIND system.
        """
        logger.info("Initializing MIND system configuration")

        # Create directories
        self.paths.ensure_directories()

        # Validate all settings
        self.validate_all()

        logger.info("MIND system configuration initialized successfully")

    def log_configuration(self) -> None:
        """Log the complete system configuration."""
        logger.info("=" * 60)
        logger.info("MIND System Configuration")
        logger.info("=" * 60)

        # Log each subsystem
        self.paths.validate_paths()  # Logs path info
        self.models.log_configuration()
        self.retrieval.log_configuration()
        self.generation.log_configuration(local_mode=self.vllm.is_local_mode())
        self.vllm.log_configuration()

        logger.info("=" * 60)

    def get_summary(self) -> str:
        """
        Get a concise summary of the configuration.

        Returns:
            Human-readable configuration summary
        """
        return (
            f"MIND Configuration:\n"
            f"  Base Dir: {self.paths.base_dir}\n"
            f"  {self.models.get_model_summary()}\n"
            f"  {self.retrieval.get_retrieval_summary()}\n"
            f"  {self.generation.get_generation_summary()}\n"
            f"  {self.vllm.get_vllm_summary()}"
        )


# Singleton pattern for global settings access
_settings_instance: Optional[MINDSettings] = None


@lru_cache(maxsize=1)
def get_settings() -> MINDSettings:
    """
    Get the global MINDSettings instance (singleton).

    This function creates and caches a single MINDSettings instance
    for the entire application. Subsequent calls return the same instance.

    Returns:
        The global MINDSettings instance

    Example:
        >>> settings = get_settings()
        >>> settings.ensure_initialized()
        >>> print(settings.paths.data_dir)
    """
    global _settings_instance

    if _settings_instance is None:
        logger.info("Creating new MINDSettings instance")
        _settings_instance = MINDSettings()

    return _settings_instance


def reload_settings() -> MINDSettings:
    """
    Reload settings from environment/files.

    Clears the cached settings instance and creates a new one.
    Use this if you need to reload configuration after environment changes.

    Returns:
        New MINDSettings instance

    Example:
        >>> import os
        >>> os.environ['MIND_VLLM_MODE'] = 'local'
        >>> settings = reload_settings()
    """
    global _settings_instance

    logger.info("Reloading settings")

    # Clear cache
    get_settings.cache_clear()
    _settings_instance = None

    # Create new instance
    return get_settings()


# Backward compatibility: Module-level constants
# These provide access to settings for legacy code
def _get_legacy_value(path: str, default=None):
    """Helper to get nested values from settings."""
    try:
        settings = get_settings()
        parts = path.split('.')
        value = settings
        for part in parts:
            value = getattr(value, part)
        return value
    except (AttributeError, TypeError):
        if default is not None:
            return default
        raise


# Export commonly used values as module constants for backward compatibility
# These will be evaluated lazily when accessed
def __getattr__(name: str):
    """
    Provide backward compatibility for module-level constants.

    This allows old code like `from config.settings import BASE_DIR`
    to continue working while using the new Pydantic-based settings.
    """
    # Path constants
    if name == "BASE_DIR":
        return _get_legacy_value("paths.base_dir")
    elif name == "DATA_DIR":
        return _get_legacy_value("paths.data_dir")
    elif name == "GUIDELINE_DIR":
        return _get_legacy_value("paths.guideline_dir")
    elif name == "PATIENT_RECORD_DIR":
        return _get_legacy_value("paths.patient_record_dir")
    elif name == "GENERATED_DOCS_DIR":
        return _get_legacy_value("paths.generated_docs_dir")
    elif name == "GUIDELINE_DB_DIR":
        return _get_legacy_value("paths.guideline_db_dir")
    elif name == "PATIENT_DB_DIR":
        return _get_legacy_value("paths.patient_db_dir")
    elif name == "GENERATED_DOCS_DB_DIR":
        return _get_legacy_value("paths.generated_docs_db_dir")
    elif name == "CACHE_DIR":
        return _get_legacy_value("paths.cache_dir")
    elif name == "PDF_FONT_PATH":
        return _get_legacy_value("paths.pdf_font_path")
    elif name == "DEFAULT_OUTPUT_NAME":
        return _get_legacy_value("paths.default_output_name")
    elif name == "DEFAULT_PATIENT_FILE":
        return _get_legacy_value("paths.default_patient_file")
    elif name == "DATE_FORMATS":
        from .paths import DATE_FORMATS as DF
        return DF

    # Model constants
    elif name == "EMBEDDING_MODEL_NAME":
        return _get_legacy_value("models.embedding_model_name")
    elif name == "RERANKER_MODEL_NAME":
        return _get_legacy_value("models.reranker_model_name")
    elif name == "EMBEDDING_DEVICE_MODE":
        return _get_legacy_value("models.embedding_device_mode")
    elif name == "RERANKER_DEVICE_MODE":
        return _get_legacy_value("models.reranker_device_mode")
    elif name == "BATCH_SIZE_RERANK":
        return _get_legacy_value("models.batch_size_rerank")

    # Retrieval constants
    elif name == "INITIAL_RETRIEVAL_K":
        return _get_legacy_value("retrieval.initial_retrieval_k")
    elif name == "FINAL_RETRIEVAL_K":
        return _get_legacy_value("retrieval.final_retrieval_k")
    elif name == "SIMILARITY_SCORE_THRESHOLD":
        return _get_legacy_value("retrieval.similarity_score_threshold")
    elif name == "GUIDELINE_SEARCH_K":
        return _get_legacy_value("retrieval.guideline_search_k")
    elif name == "GENERATED_DOC_SEARCH_K":
        return _get_legacy_value("retrieval.generated_doc_search_k")
    elif name == "USE_HYBRID_MULTI_FACT_APPROACH":
        return _get_legacy_value("retrieval.use_hybrid_multi_fact_approach")
    elif name == "MAX_SOURCES_PER_FACT":
        return _get_legacy_value("retrieval.max_sources_per_fact")

    # Generation constants
    elif name == "TEMPERATURE":
        return _get_legacy_value("generation.temperature")
    elif name == "TOP_P":
        return _get_legacy_value("generation.top_p")
    elif name == "TOP_K":
        return _get_legacy_value("generation.top_k")
    elif name == "MIN_P":
        return _get_legacy_value("generation.min_p")
    elif name == "PRESENCE_PENALTY":
        return _get_legacy_value("generation.presence_penalty")
    elif name == "DEFAULT_VALIDATION_CYCLES":
        return _get_legacy_value("generation.default_validation_cycles")
    elif name == "MAX_VALIDATION_CYCLES":
        return _get_legacy_value("generation.max_validation_cycles")
    elif name == "MIN_VALIDATION_CYCLES":
        return _get_legacy_value("generation.min_validation_cycles")
    elif name == "MAX_AGENT_ITERATIONS":
        return _get_legacy_value("generation.max_agent_iterations")
    elif name == "AGENT_EARLY_STOPPING_METHOD":
        return _get_legacy_value("generation.agent_early_stopping_method")
    elif name == "MEMORY_MAX_TOKEN_LIMIT":
        return _get_legacy_value("generation.memory_max_token_limit")
    elif name == "CHAT_HISTORY_LIMIT":
        return _get_legacy_value("generation.chat_history_limit")

    # vLLM constants
    elif name == "VLLM_MODE":
        return _get_legacy_value("vllm.vllm_mode")
    elif name == "VLLM_SERVER_URL":
        return _get_legacy_value("vllm.vllm_server_url")
    elif name == "VLLM_MODEL_NAME":
        return _get_legacy_value("vllm.vllm_model_name")
    elif name == "VLLM_GPU_MEMORY_UTILIZATION":
        return _get_legacy_value("vllm.vllm_gpu_memory_utilization")
    elif name == "VLLM_MAX_MODEL_LEN":
        return _get_legacy_value("vllm.vllm_max_model_len")
    elif name == "VLLM_MAX_NUM_SEQS":
        return _get_legacy_value("vllm.vllm_max_num_seqs")

    # Legacy functions
    elif name == "ensure_directories":
        return lambda: get_settings().paths.ensure_directories()
    elif name == "get_patient_file_path":
        return lambda: get_settings().paths.get_patient_file_path()

    # Special handling for device config (removed - now handled by core/device_manager)
    elif name == "get_device_config":
        logger.warning(
            "get_device_config() is deprecated and has been moved to "
            "core/device_manager.py. Please update your imports."
        )
        # Return a dummy function that raises an informative error
        def _deprecated_get_device_config():
            raise NotImplementedError(
                "get_device_config() has been moved to core/device_manager.py. "
                "Import from there instead: "
                "from core.device_manager import get_device_config"
            )
        return _deprecated_get_device_config

    # Legacy device constants (will be set by core/device_manager)
    elif name in ("EMBEDDING_DEVICE", "RERANKER_DEVICE", "DEVICE_INFO"):
        logger.warning(
            f"{name} is no longer set in config.settings. "
            "Device configuration is now handled by core/device_manager.py"
        )
        return None

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
