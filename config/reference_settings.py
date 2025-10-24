"""
Reference tracking configuration for the medical documentation system.
Controls how source references are included in generated documents.
"""

# Default reference settings
DEFAULT_INCLUDE_REFERENCES = True
DEFAULT_MAX_REFERENCES_PER_SECTION = 3
DEFAULT_SHOW_REFERENCE_STATISTICS = True

# Reference formatting options
REFERENCE_FORMAT_INLINE = "[Kilde: {entry_type} - {timestamp}]"
REFERENCE_FORMAT_DETAILED = "[Kilde: {entry_type} - {timestamp} (Relevans: {relevance}%)]"

# Quality thresholds for reference inclusion
MIN_RELEVANCE_THRESHOLD = 60  # Only include references with ≥60% relevance
PREFER_RECENT_SOURCES = True  # Prioritize newer sources when available
MAX_DAYS_OLD_PREFERRED = 30   # Prefer sources from last 30 days

# Reference grouping settings
GROUP_REFERENCES_BY_TYPE = True
DEDUPLICATE_SAME_TIMESTAMP = True
MAX_REFERENCES_IN_APPENDIX = 50

# Output formatting
INCLUDE_REFERENCE_APPENDIX = True
INCLUDE_SOURCE_STATISTICS = True
INCLUDE_QUALITY_INDICATORS = True

class ReferenceConfig:
    """Configuration class for reference tracking settings."""
    
    def __init__(self):
        self.include_references = DEFAULT_INCLUDE_REFERENCES
        self.max_references_per_section = DEFAULT_MAX_REFERENCES_PER_SECTION
        self.show_statistics = DEFAULT_SHOW_REFERENCE_STATISTICS
        self.reference_format = REFERENCE_FORMAT_INLINE
        self.min_relevance = MIN_RELEVANCE_THRESHOLD
        self.prefer_recent = PREFER_RECENT_SOURCES
        self.max_days_preferred = MAX_DAYS_OLD_PREFERRED
        self.group_by_type = GROUP_REFERENCES_BY_TYPE
        self.deduplicate = DEDUPLICATE_SAME_TIMESTAMP
        self.include_appendix = INCLUDE_REFERENCE_APPENDIX
        self.include_stats = INCLUDE_SOURCE_STATISTICS
    
    def get_reference_settings(self) -> dict:
        """Get current reference settings as dictionary."""
        return {
            "include_references": self.include_references,
            "max_references_per_section": self.max_references_per_section,
            "show_statistics": self.show_statistics,
            "reference_format": self.reference_format,
            "min_relevance_threshold": self.min_relevance,
            "prefer_recent_sources": self.prefer_recent,
            "max_days_old_preferred": self.max_days_preferred,
            "group_references_by_type": self.group_by_type,
            "deduplicate_same_timestamp": self.deduplicate,
            "include_reference_appendix": self.include_appendix,
            "include_source_statistics": self.include_stats
        }
    
    def set_high_detail_mode(self):
        """Configure for maximum reference detail."""
        self.max_references_per_section = 5
        self.reference_format = REFERENCE_FORMAT_DETAILED
        self.min_relevance = 50
        self.include_appendix = True
        self.include_stats = True
        
    def set_minimal_mode(self):
        """Configure for minimal reference inclusion."""
        self.max_references_per_section = 1
        self.reference_format = REFERENCE_FORMAT_INLINE
        self.min_relevance = 80
        self.include_appendix = False
        self.include_stats = False
        
    def set_balanced_mode(self):
        """Configure for balanced reference inclusion (default)."""
        self.max_references_per_section = 3
        self.reference_format = REFERENCE_FORMAT_INLINE
        self.min_relevance = 60
        self.include_appendix = True
        self.include_stats = True

# Global configuration instance
reference_config = ReferenceConfig()

# Preset configurations
REFERENCE_PRESETS = {
    "minimal": {
        "include_references": True,
        "max_references_per_section": 1,
        "min_relevance": 80,
        "include_appendix": False,
        "description": "Minimal references - only the most relevant source per section"
    },
    "balanced": {
        "include_references": True, 
        "max_references_per_section": 3,
        "min_relevance": 60,
        "include_appendix": True,
        "description": "Balanced references - 3 sources per section with appendix"
    },
    "detailed": {
        "include_references": True,
        "max_references_per_section": 5,
        "min_relevance": 50,
        "include_appendix": True,
        "description": "Detailed references - up to 5 sources per section with full appendix"
    },
    "none": {
        "include_references": False,
        "max_references_per_section": 0,
        "min_relevance": 100,
        "include_appendix": False,
        "description": "No references - original document generation without citations"
    }
}

def get_preset_config(preset_name: str) -> dict:
    """Get configuration for a specific preset."""
    return REFERENCE_PRESETS.get(preset_name.lower(), REFERENCE_PRESETS["balanced"])

def apply_preset(preset_name: str) -> None:
    """Apply a preset configuration to the global config."""
    preset = get_preset_config(preset_name)
    
    reference_config.include_references = preset["include_references"]
    reference_config.max_references_per_section = preset["max_references_per_section"] 
    reference_config.min_relevance = preset["min_relevance"]
    reference_config.include_appendix = preset["include_appendix"]
    
    print(f"[CONFIG] Applied '{preset_name}' preset: {preset['description']}")