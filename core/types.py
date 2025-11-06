"""
Type definitions and aliases for the MIND core module.

This module provides common type aliases used throughout the core infrastructure
to improve type safety and code readability.

Type Categories:
    - Device types: GPU/CPU device identifiers
    - Memory types: Memory measurements and statistics
    - Database types: Database statistics and metadata

Example:
    >>> from core.types import DeviceType, MemoryInfo
    >>> def allocate_model(device: DeviceType) -> None:
    ...     pass
"""

from typing import Dict, Literal
from typing_extensions import TypeAlias

# Device type identifiers
DeviceType: TypeAlias = Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]

# Memory information dictionary with standardized keys
MemoryInfo: TypeAlias = Dict[str, float]  # Keys: 'total_gb', 'used_gb', 'free_gb'

# Database statistics dictionary
DatabaseStats: TypeAlias = Dict[str, int]  # Keys: database names, values: document counts

# Model configuration dictionary
ModelConfig: TypeAlias = Dict[str, any]  # Model configuration parameters
