"""Memory estimation utilities for DataChain."""

import sys
from typing import Any, Union

# Default batch processing values
DEFAULT_CHUNK_ROWS = 2000
DEFAULT_CHUNK_MB = 1000

# Memory monitoring threshold (percentage)
MEMORY_USAGE_THRESHOLD = 80

# System memory check frequency (every N rows)
MEMORY_CHECK_FREQUENCY = 100

# Shared constant for object overhead estimation
OBJECT_OVERHEAD_BYTES = 100


def estimate_memory_recursive(item: Any) -> int:
    if item is None:
        return 0

    if isinstance(item, (str, bytes, int, float, bool)):
        return sys.getsizeof(item)
    if isinstance(item, (list, tuple)):
        total_size = sys.getsizeof(item)
        for subitem in item:
            total_size += sys.getsizeof(subitem)
        return total_size
    # For complex objects, use a conservative estimate
    return sys.getsizeof(item) + OBJECT_OVERHEAD_BYTES


def estimate_row_memory(row: Union[list, tuple]) -> int:
    if not row:
        return 0

    total_size = 0
    for item in row:
        total_size += estimate_memory_recursive(item)

    return total_size


def get_system_memory_percent() -> float:
    try:
        import psutil

        return psutil.virtual_memory().percent
    except ImportError:
        import warnings

        warnings.warn(
            "psutil not available. Memory-based checks will be skipped. "
            "Install psutil to enable memory monitoring.",
            UserWarning,
            stacklevel=2,
        )
        return 0.0


def is_memory_usage_high() -> bool:
    return get_system_memory_percent() > MEMORY_USAGE_THRESHOLD
