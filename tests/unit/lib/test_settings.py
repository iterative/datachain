import pytest

from datachain.lib.settings import Settings, SettingsError
from datachain.query.batch import DynamicBatch


def test_settings_defaults_and_custom():
    """Test Settings class with default and custom batch parameters."""
    # Default values
    settings = Settings()
    assert settings.batch_rows == 2000
    assert settings.batch_mem == 1000

    # Custom values
    settings = Settings(batch_rows=500, batch_mem=750.5)
    assert settings.batch_rows == 500
    assert settings.batch_mem == 750.5

    # to_dict method
    d = settings.to_dict()
    assert d["batch_rows"] == 500
    assert d["batch_mem"] == 750.5

    # Chaining
    s2 = settings
    s3 = s2
    assert s3.batch_rows == 500
    assert s3.batch_mem == 750.5


def test_settings_validation():
    # Valid
    settings = Settings(batch_rows=100, batch_mem=50.5)
    assert settings.batch_rows == 100
    assert settings.batch_mem == 50.5

    # Invalid batch_rows
    with pytest.raises(SettingsError):
        Settings(batch_rows="invalid")

    # Invalid batch_mem
    with pytest.raises(SettingsError):
        Settings(batch_mem="invalid")


def test_dynamic_batch_memory_monitoring():
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
    estimated_memory = dynamic_batch._estimate_row_memory(test_row)
    assert estimated_memory > 0

    # Test with empty row
    empty_memory = dynamic_batch._estimate_row_memory([])
    assert empty_memory == 0
