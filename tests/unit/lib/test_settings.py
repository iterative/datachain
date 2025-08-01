import pytest

from datachain.lib.memory_utils import estimate_row_memory
from datachain.lib.settings import Settings, SettingsError
from datachain.query.batch import DynamicBatch


def test_settings_defaults_and_custom():
    """Test Settings class with default and custom batch parameters."""
    # Default values
    settings = Settings()
    assert settings.chunk_rows == 2000
    assert settings.chunk_mb == 1000

    # Custom values
    settings = Settings(chunk_rows=500, chunk_mb=750.5)
    assert settings.chunk_rows == 500
    assert settings.chunk_mb == 750.5

    # to_dict method
    d = settings.to_dict()
    assert d["chunk_rows"] == 500
    assert d["chunk_mb"] == 750.5

    # Chaining
    s2 = settings
    s3 = s2
    assert s3.chunk_rows == 500
    assert s3.chunk_mb == 750.5


def test_settings_validation():
    # Valid
    settings = Settings(chunk_rows=100, chunk_mb=50.5)
    assert settings.chunk_rows == 100
    assert settings.chunk_mb == 50.5

    # Invalid chunk_rows
    with pytest.raises(SettingsError):
        Settings(chunk_rows="invalid")

    # Invalid chunk_mb
    with pytest.raises(SettingsError):
        Settings(chunk_mb="invalid")

    # Zero chunk_rows
    with pytest.raises(SettingsError):
        Settings(chunk_rows=0)

    # Zero chunk_mb
    with pytest.raises(SettingsError):
        Settings(chunk_mb=0)


def test_dynamic_chunk_memory_monitoring():
    """Test that DynamicBatch integrates memory monitoring correctly."""
    # Create a DynamicBatch instance
    dynamic_batch = DynamicBatch(max_rows=100, max_memory_mb=50)

    # Test that it has the expected attributes
    assert dynamic_batch.max_rows == 100
    assert dynamic_batch.max_memory_mb == 50
    assert dynamic_batch.max_memory_bytes == 50 * 1024 * 1024
    assert dynamic_batch.is_batching is True

    # Test memory estimation
    test_row = [1, "test", [1, 2, 3]]
    estimated_memory = estimate_row_memory(test_row)
    assert estimated_memory > 0

    # Test with empty row
    empty_memory = estimate_row_memory([])
    assert empty_memory == 0


def test_settings_add():
    """Test Settings.add() method with chunk_rows and chunk_mb."""
    # Create base settings
    base_settings = Settings(chunk_rows=1000, chunk_mb=500)

    # Create settings to add
    add_settings = Settings(chunk_rows=2000, chunk_mb=750)

    # Add settings
    base_settings.add(add_settings)

    # Verify that values from add_settings override base_settings
    assert base_settings.chunk_rows == 2000
    assert base_settings.chunk_mb == 750

    # Test with None values (should not override)
    base_settings = Settings(chunk_rows=1000, chunk_mb=500)
    none_settings = Settings(chunk_rows=None, chunk_mb=None)

    base_settings.add(none_settings)

    # Verify that None values don't override existing values
    assert base_settings.chunk_rows == 1000
    assert base_settings.chunk_mb == 500
