import pytest

from datachain.lib.settings import Settings, SettingsError


def test_settings_defaults_and_custom():
    """Test Settings class with default and custom batch parameters."""
    # Default values
    settings = Settings()
    assert settings.chunk_rows == 2000

    # Custom values
    settings = Settings(chunk_rows=500)
    assert settings.chunk_rows == 500

    # to_dict method
    d = settings.to_dict()
    assert d["chunk_rows"] == 500

    # Chaining
    s2 = settings
    s3 = s2
    assert s3.chunk_rows == 500


def test_settings_validation():
    # Valid
    settings = Settings(chunk_rows=100)
    assert settings.chunk_rows == 100

    # Invalid chunk_rows
    with pytest.raises(SettingsError):
        Settings(chunk_rows="invalid")

    # Zero chunk_rows
    with pytest.raises(SettingsError):
        Settings(chunk_rows=0)


def test_settings_add():
    """Test Settings.add() method with chunk_rows and chunk_mb."""
    # Create base settings
    base_settings = Settings(chunk_rows=1000)

    # Create settings to add
    add_settings = Settings(chunk_rows=2000)

    # Add settings
    base_settings.add(add_settings)

    # Verify that values from add_settings override base_settings
    assert base_settings.chunk_rows == 2000

    # Test with None values (should not override)
    base_settings = Settings(chunk_rows=1000)
    none_settings = Settings(chunk_rows=None)

    base_settings.add(none_settings)

    # Verify that None values don't override existing values
    assert base_settings.chunk_rows == 1000
