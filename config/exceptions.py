"""
Custom exception classes for configuration-related errors.

This module defines domain-specific exceptions used throughout the
configuration system to provide clear error messaging and error handling.
"""


class ConfigurationError(Exception):
    """Base exception for all configuration-related errors."""
    pass


class InvalidConfigValueError(ConfigurationError):
    """Raised when a configuration value is invalid or out of range."""
    pass


class MissingRequiredConfigError(ConfigurationError):
    """Raised when a required configuration parameter is missing."""
    pass


class InvalidPathError(ConfigurationError):
    """Raised when a path configuration is invalid or inaccessible."""
    pass


class InvalidDeviceModeError(ConfigurationError):
    """Raised when an invalid device mode is specified."""
    pass


class VLLMConfigurationError(ConfigurationError):
    """Raised when vLLM configuration is invalid."""
    pass


class ModelConfigurationError(ConfigurationError):
    """Raised when model configuration is invalid."""
    pass
