"""
Configuration package for the MIND medical documentation system.

Provides centralized, validated configuration management using Pydantic.
All configuration can be overridden via environment variables (MIND_* prefix)
or .env file.

Quick Start:
    >>> from config import get_settings
    >>> settings = get_settings()
    >>> settings.ensure_initialized()
    >>> print(settings.get_summary())

Module Structure:
    - settings: Main MINDSettings aggregator with all subsystems
    - paths: Directory and file path configuration
    - models: ML model configuration (embeddings, reranker)
    - retrieval: RAG and search parameter configuration
    - generation: Document generation and LLM sampling configuration
    - vllm: vLLM server/local mode configuration
    - llm_config: LLM client management
    - reference_settings: Reference tracking configuration
    - exceptions: Custom exception classes
"""

# Main settings class and factory
from .settings import (
    MINDSettings,
    get_settings,
    reload_settings,
)

# Configuration subsystems
from .paths import PathConfig, DATE_FORMATS
from .models import ModelConfig, DeviceMode
from .retrieval import RetrievalConfig
from .generation import GenerationConfig
from .vllm import VLLMConfig, VLLMMode

# LLM client configuration
from .llm_config import (
    LLMConfig,
    VLLMClient,
    InPythonVLLMClient,
    llm_config,
    get_llm_config,
)

# Reference tracking
from .reference_settings import (
    ReferenceConfig,
    reference_config,
    ReferencePresetName,
    REFERENCE_PRESETS,
    get_preset_config,
    apply_preset,
    # Constants for backward compatibility
    DEFAULT_INCLUDE_REFERENCES,
    DEFAULT_MAX_REFERENCES_PER_SECTION,
    DEFAULT_SHOW_REFERENCE_STATISTICS,
    REFERENCE_FORMAT_INLINE,
    REFERENCE_FORMAT_DETAILED,
    MIN_RELEVANCE_THRESHOLD,
    PREFER_RECENT_SOURCES,
    MAX_DAYS_OLD_PREFERRED,
    GROUP_REFERENCES_BY_TYPE,
    DEDUPLICATE_SAME_TIMESTAMP,
    MAX_REFERENCES_IN_APPENDIX,
    INCLUDE_REFERENCE_APPENDIX,
    INCLUDE_SOURCE_STATISTICS,
    INCLUDE_QUALITY_INDICATORS,
)

# Exceptions
from .exceptions import (
    ConfigurationError,
    InvalidConfigValueError,
    MissingRequiredConfigError,
    InvalidPathError,
    InvalidDeviceModeError,
    VLLMConfigurationError,
    ModelConfigurationError,
)

# For backward compatibility, provide easy access to module-level constants
# These delegate to the settings instance
__all__ = [
    # Main settings
    'MINDSettings',
    'get_settings',
    'reload_settings',

    # Configuration subsystems
    'PathConfig',
    'ModelConfig',
    'RetrievalConfig',
    'GenerationConfig',
    'VLLMConfig',

    # LLM configuration
    'LLMConfig',
    'VLLMClient',
    'InPythonVLLMClient',
    'llm_config',
    'get_llm_config',

    # Reference configuration
    'ReferenceConfig',
    'reference_config',
    'ReferencePresetName',
    'REFERENCE_PRESETS',
    'get_preset_config',
    'apply_preset',

    # Type aliases
    'DeviceMode',
    'VLLMMode',

    # Constants
    'DATE_FORMATS',

    # Reference constants
    'DEFAULT_INCLUDE_REFERENCES',
    'DEFAULT_MAX_REFERENCES_PER_SECTION',
    'DEFAULT_SHOW_REFERENCE_STATISTICS',
    'REFERENCE_FORMAT_INLINE',
    'REFERENCE_FORMAT_DETAILED',
    'MIN_RELEVANCE_THRESHOLD',
    'PREFER_RECENT_SOURCES',
    'MAX_DAYS_OLD_PREFERRED',
    'GROUP_REFERENCES_BY_TYPE',
    'DEDUPLICATE_SAME_TIMESTAMP',
    'MAX_REFERENCES_IN_APPENDIX',
    'INCLUDE_REFERENCE_APPENDIX',
    'INCLUDE_SOURCE_STATISTICS',
    'INCLUDE_QUALITY_INDICATORS',

    # Exceptions
    'ConfigurationError',
    'InvalidConfigValueError',
    'MissingRequiredConfigError',
    'InvalidPathError',
    'InvalidDeviceModeError',
    'VLLMConfigurationError',
    'ModelConfigurationError',
]


# Provide backward compatibility for old import patterns
# e.g., `from config import BASE_DIR`
def __getattr__(name: str):
    """
    Provide backward compatibility for module-level imports.

    Allows old code like `from config import BASE_DIR` to continue working
    by delegating to the settings instance.
    """
    # Try to get from settings module first (which has __getattr__ for legacy support)
    from . import settings as settings_module

    try:
        return getattr(settings_module, name)
    except AttributeError:
        pass

    # If not found, raise AttributeError
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
