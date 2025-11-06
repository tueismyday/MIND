"""
Reference tracking configuration for the MIND medical documentation system.

Controls how source references are included in generated documents,
including formatting, quality thresholds, and preset modes.
"""

import logging
from typing import Dict, Any, Literal

from pydantic import BaseModel, Field, field_validator

from .exceptions import InvalidConfigValueError

logger = logging.getLogger(__name__)

# Type aliases
ReferencePresetName = Literal["minimal", "balanced", "detailed", "none"]


class ReferenceConfig(BaseModel):
    """
    Configuration for reference tracking and formatting in generated documents.

    Controls how source references (citations to patient records and guidelines)
    are included, formatted, and displayed in generated medical documents.

    Attributes:
        include_references: Whether to include references in documents
        max_references_per_section: Maximum number of references per section
        show_statistics: Show reference statistics in output
        reference_format: Format string for references
        min_relevance: Minimum relevance score (0-100) for reference inclusion
        prefer_recent: Prioritize newer sources
        max_days_preferred: Consider sources within this many days as "recent"
        group_by_type: Group references by entry type
        deduplicate: Remove duplicate references with same timestamp
        include_appendix: Include reference appendix at document end
        include_stats: Include source statistics summary

    Example:
        >>> config = ReferenceConfig()
        >>> config.set_balanced_mode()
        >>> print(config.get_reference_settings())
    """

    # Core settings
    include_references: bool = Field(
        default=True,
        description="Include source references in generated documents"
    )

    max_references_per_section: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of references to include per section"
    )

    show_statistics: bool = Field(
        default=True,
        description="Show reference statistics in output"
    )

    # Formatting
    reference_format: str = Field(
        default="[Kilde: {entry_type} - {timestamp}]",
        description="Format string for inline references"
    )

    # Quality thresholds
    min_relevance: int = Field(
        default=60,
        ge=0,
        le=100,
        description="Minimum relevance score (0-100) for reference inclusion"
    )

    prefer_recent: bool = Field(
        default=True,
        description="Prioritize newer sources when available"
    )

    max_days_preferred: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Max age in days for a source to be considered recent"
    )

    # Grouping and deduplication
    group_by_type: bool = Field(
        default=True,
        description="Group references by entry type in appendix"
    )

    deduplicate: bool = Field(
        default=True,
        description="Remove duplicate references with same timestamp"
    )

    # Output options
    include_appendix: bool = Field(
        default=True,
        description="Include reference appendix at end of document"
    )

    include_stats: bool = Field(
        default=True,
        description="Include source statistics summary"
    )

    model_config = {
        "validate_assignment": True
    }

    @field_validator('min_relevance')
    @classmethod
    def validate_min_relevance(cls, v: int) -> int:
        """Validate minimum relevance is reasonable."""
        if v < 0 or v > 100:
            raise InvalidConfigValueError(
                f"min_relevance must be between 0 and 100, got {v}"
            )
        if v > 90:
            logger.warning(
                f"Very high minimum relevance ({v}%). "
                "This may exclude most references."
            )
        return v

    @field_validator('max_references_per_section')
    @classmethod
    def validate_max_references(cls, v: int) -> int:
        """Validate max references per section is reasonable."""
        if v < 0:
            raise InvalidConfigValueError(
                f"max_references_per_section must be non-negative, got {v}"
            )
        if v > 10:
            logger.warning(
                f"Large number of references per section ({v}). "
                "This may clutter the document."
            )
        return v

    def get_reference_settings(self) -> Dict[str, Any]:
        """
        Get current reference settings as dictionary.

        Returns:
            Dictionary of all reference configuration settings
        """
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

    def set_high_detail_mode(self) -> None:
        """Configure for maximum reference detail."""
        self.max_references_per_section = 5
        self.reference_format = "[Kilde: {entry_type} - {timestamp} (Relevans: {relevance}%)]"
        self.min_relevance = 50
        self.include_appendix = True
        self.include_stats = True
        logger.info("Reference mode set to: High Detail")

    def set_minimal_mode(self) -> None:
        """Configure for minimal reference inclusion."""
        self.max_references_per_section = 1
        self.reference_format = "[Kilde: {entry_type} - {timestamp}]"
        self.min_relevance = 80
        self.include_appendix = False
        self.include_stats = False
        logger.info("Reference mode set to: Minimal")

    def set_balanced_mode(self) -> None:
        """Configure for balanced reference inclusion (default)."""
        self.max_references_per_section = 3
        self.reference_format = "[Kilde: {entry_type} - {timestamp}]"
        self.min_relevance = 60
        self.include_appendix = True
        self.include_stats = True
        logger.info("Reference mode set to: Balanced")

    def set_no_references_mode(self) -> None:
        """Disable all references."""
        self.include_references = False
        self.max_references_per_section = 0
        self.min_relevance = 100
        self.include_appendix = False
        logger.info("Reference mode set to: None")

    def apply_preset(self, preset_name: ReferencePresetName) -> None:
        """
        Apply a preset configuration.

        Args:
            preset_name: Name of the preset ("minimal", "balanced", "detailed", "none")

        Raises:
            InvalidConfigValueError: If preset name is invalid
        """
        if preset_name == "minimal":
            self.set_minimal_mode()
        elif preset_name == "balanced":
            self.set_balanced_mode()
        elif preset_name == "detailed":
            self.set_high_detail_mode()
        elif preset_name == "none":
            self.set_no_references_mode()
        else:
            raise InvalidConfigValueError(
                f"Invalid preset name: {preset_name}. "
                f"Must be one of: minimal, balanced, detailed, none"
            )

    def log_configuration(self) -> None:
        """Log the current reference configuration."""
        logger.info("Reference Configuration:")
        logger.info(f"  Include References: {self.include_references}")
        if self.include_references:
            logger.info(f"  Max Per Section: {self.max_references_per_section}")
            logger.info(f"  Min Relevance: {self.min_relevance}%")
            logger.info(f"  Prefer Recent: {self.prefer_recent}")
            logger.info(f"  Include Appendix: {self.include_appendix}")
            logger.info(f"  Include Statistics: {self.include_stats}")


# Global configuration instance
reference_config = ReferenceConfig()


# Preset configurations as constants for backward compatibility
REFERENCE_PRESETS: Dict[str, Dict[str, Any]] = {
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


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """
    Get configuration dictionary for a specific preset.

    Args:
        preset_name: Name of the preset

    Returns:
        Dictionary containing preset configuration
    """
    return REFERENCE_PRESETS.get(
        preset_name.lower(),
        REFERENCE_PRESETS["balanced"]
    )


def apply_preset(preset_name: str) -> None:
    """
    Apply a preset configuration to the global config.

    Args:
        preset_name: Name of the preset to apply

    Example:
        >>> apply_preset("minimal")
        >>> print(reference_config.max_references_per_section)
        1
    """
    reference_config.apply_preset(preset_name.lower())  # type: ignore


# Module-level constants for backward compatibility
DEFAULT_INCLUDE_REFERENCES = True
DEFAULT_MAX_REFERENCES_PER_SECTION = 3
DEFAULT_SHOW_REFERENCE_STATISTICS = True

# Reference formatting options
REFERENCE_FORMAT_INLINE = "[Kilde: {entry_type} - {timestamp}]"
REFERENCE_FORMAT_DETAILED = "[Kilde: {entry_type} - {timestamp} (Relevans: {relevance}%)]"

# Quality thresholds
MIN_RELEVANCE_THRESHOLD = 60
PREFER_RECENT_SOURCES = True
MAX_DAYS_OLD_PREFERRED = 30

# Reference grouping settings
GROUP_REFERENCES_BY_TYPE = True
DEDUPLICATE_SAME_TIMESTAMP = True
MAX_REFERENCES_IN_APPENDIX = 50

# Output formatting
INCLUDE_REFERENCE_APPENDIX = True
INCLUDE_SOURCE_STATISTICS = True
INCLUDE_QUALITY_INDICATORS = True
