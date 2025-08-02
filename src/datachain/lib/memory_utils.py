"""Memory estimation utilities for DataChain."""

# Default batch processing values
DEFAULT_CHUNK_ROWS = 2000

# Memory monitoring threshold (percentage)
MEMORY_USAGE_THRESHOLD = 80


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
