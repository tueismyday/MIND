"""
Configuration package for the Agentic RAG Medical Documentation System.
Provides centralized access to settings, LLM configuration, and reference tracking.
"""

# Core settings
from .settings import (
    # Directories
    BASE_DIR,
    DATA_DIR,
    GUIDELINE_DIR,
    PATIENT_RECORD_DIR,
    GENERATED_DOCS_DIR,
    GUIDELINE_DB_DIR,
    PATIENT_DB_DIR,
    GENERATED_DOCS_DB_DIR,
    CACHE_DIR,
    
    # Model configuration
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    
    # Agent configuration
    MAX_AGENT_ITERATIONS,
    AGENT_EARLY_STOPPING_METHOD,
    MEMORY_MAX_TOKEN_LIMIT,
    CHAT_HISTORY_LIMIT,
    
    # Retrieval configuration
    INITIAL_RETRIEVAL_K,
    FINAL_RETRIEVAL_K,
    SIMILARITY_SCORE_THRESHOLD,
    GUIDELINE_SEARCH_K,
    GENERATED_DOC_SEARCH_K,
    PATIENT_SEARCH_K,
    
    # Generation configuration
    GENERATION_TEMPERATURE,
    CRITIQUE_TEMPERATURE,
    
    # Validation configuration
    DEFAULT_VALIDATION_CYCLES,
    MAX_VALIDATION_CYCLES,
    MIN_VALIDATION_CYCLES,
    
    # Hybrid approach configuration
    USE_HYBRID_MULTI_FACT_APPROACH,
    FACT_COMPLEXITY_THRESHOLD,
    MAX_SOURCES_PER_FACT,
    
    # PDF configuration
    PDF_FONT_PATH,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PATIENT_FILE,

    # Date formats
    DATE_FORMATS,

    # Utility functions
    ensure_directories,
    get_patient_file_path,
)

# LLM configuration
from .llm_config import (
    LLMConfig,
    VLLMClient,
    llm_config,  # Global instance
)

# Reference tracking configuration
from .reference_settings import (
    # Default settings
    DEFAULT_INCLUDE_REFERENCES,
    DEFAULT_MAX_REFERENCES_PER_SECTION,
    DEFAULT_SHOW_REFERENCE_STATISTICS,
    
    # Reference formats
    REFERENCE_FORMAT_INLINE,
    REFERENCE_FORMAT_DETAILED,
    
    # Quality thresholds
    MIN_RELEVANCE_THRESHOLD,
    PREFER_RECENT_SOURCES,
    MAX_DAYS_OLD_PREFERRED,
    
    # Grouping settings
    GROUP_REFERENCES_BY_TYPE,
    DEDUPLICATE_SAME_TIMESTAMP,
    MAX_REFERENCES_IN_APPENDIX,
    
    # Output formatting
    INCLUDE_REFERENCE_APPENDIX,
    INCLUDE_SOURCE_STATISTICS,
    INCLUDE_QUALITY_INDICATORS,
    
    # Classes and functions
    ReferenceConfig,
    reference_config,  # Global instance
    REFERENCE_PRESETS,
    get_preset_config,
    apply_preset,
)

__all__ = [
    # Settings
    'BASE_DIR', 'DATA_DIR', 'GUIDELINE_DIR', 'PATIENT_RECORD_DIR',
    'GENERATED_DOCS_DIR', 'GUIDELINE_DB_DIR', 'PATIENT_DB_DIR',
    'GENERATED_DOCS_DB_DIR', 'CACHE_DIR',
    'EMBEDDING_MODEL_NAME', 'EMBEDDING_DEVICE',
    'MAX_AGENT_ITERATIONS', 'AGENT_EARLY_STOPPING_METHOD',
    'MEMORY_MAX_TOKEN_LIMIT', 'CHAT_HISTORY_LIMIT',
    'INITIAL_RETRIEVAL_K', 'FINAL_RETRIEVAL_K', 'SIMILARITY_SCORE_THRESHOLD',
    'GUIDELINE_SEARCH_K', 'GENERATED_DOC_SEARCH_K', 'PATIENT_SEARCH_K',
    'GENERATION_TEMPERATURE', 'CRITIQUE_TEMPERATURE',
    'DEFAULT_VALIDATION_CYCLES', 'MAX_VALIDATION_CYCLES', 'MIN_VALIDATION_CYCLES',
    'USE_HYBRID_MULTI_FACT_APPROACH', 'FACT_COMPLEXITY_THRESHOLD', 'MAX_SOURCES_PER_FACT',
    'PDF_FONT_PATH', 'DEFAULT_OUTPUT_NAME', 'DEFAULT_PATIENT_FILE',
    'DATE_FORMATS',
    'ensure_directories', 'get_patient_file_path',
    
    # LLM
    'LLMConfig', 'VLLMClient', 'llm_config',
    
    # References
    'DEFAULT_INCLUDE_REFERENCES', 'DEFAULT_MAX_REFERENCES_PER_SECTION',
    'DEFAULT_SHOW_REFERENCE_STATISTICS',
    'REFERENCE_FORMAT_INLINE', 'REFERENCE_FORMAT_DETAILED',
    'MIN_RELEVANCE_THRESHOLD', 'PREFER_RECENT_SOURCES', 'MAX_DAYS_OLD_PREFERRED',
    'GROUP_REFERENCES_BY_TYPE', 'DEDUPLICATE_SAME_TIMESTAMP', 'MAX_REFERENCES_IN_APPENDIX',
    'INCLUDE_REFERENCE_APPENDIX', 'INCLUDE_SOURCE_STATISTICS', 'INCLUDE_QUALITY_INDICATORS',
    'ReferenceConfig', 'reference_config', 'REFERENCE_PRESETS',
    'get_preset_config', 'apply_preset',
]