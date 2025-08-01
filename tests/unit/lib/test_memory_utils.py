"""Tests for memory utility functions."""

from datachain.lib.memory_utils import (
    OBJECT_OVERHEAD_BYTES,
    estimate_memory_recursive,
    estimate_row_memory,
    get_system_memory_percent,
)


def test_estimate_memory_recursive():
    """Test memory estimation for basic types."""
    assert estimate_memory_recursive(None) == 0
    assert estimate_memory_recursive(42) > 0
    assert estimate_memory_recursive("test") > 0
    assert estimate_memory_recursive([1, 2, 3]) > 0


def test_estimate_row_memory():
    """Test memory estimation for rows."""
    assert estimate_row_memory([]) == 0
    assert estimate_row_memory([1, "test", 3.14]) > 0


def test_system_memory_functions():
    """Test system memory monitoring functions."""
    memory_percent = get_system_memory_percent()
    assert isinstance(memory_percent, (int, float))
    assert 0.0 <= memory_percent <= 100.0


def test_object_overhead_constant():
    """Test that OBJECT_OVERHEAD_BYTES is defined."""
    assert isinstance(OBJECT_OVERHEAD_BYTES, int)
    assert OBJECT_OVERHEAD_BYTES > 0
