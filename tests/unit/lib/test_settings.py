import pytest

from datachain.lib.settings import Settings, SettingsError


def test_settings_defaults_and_custom():
    """Test Settings class with default and custom batch parameters."""
    # Default values
    settings = Settings()
    assert settings.batch_rows == 2000
    assert settings.batch_mb is None

    # Custom values
    settings = Settings(batch_rows=500, batch_mb=100)
    assert settings.batch_rows == 500
    assert settings.batch_mb == 100

    # to_dict method
    d = settings.to_dict()
    assert d["batch_rows"] == 500
    assert d["batch_mb"] == 100

    # Chaining
    s2 = settings
    s3 = s2
    assert s3.batch_rows == 500
    assert s3.batch_mb == 100


def test_settings_validation():
    # Valid
    settings = Settings(batch_rows=100, batch_mb=100)
    assert settings.batch_rows == 100
    assert settings.batch_mb == 100

    # Invalid batch_rows
    with pytest.raises(SettingsError):
        Settings(batch_rows="invalid")

    # Zero batch_rows
    with pytest.raises(SettingsError):
        Settings(batch_rows=0)

    # Invalid batch_mb
    with pytest.raises(SettingsError):
        Settings(batch_mb="invalid")

    # Zero batch_mb
    with pytest.raises(SettingsError):
        Settings(batch_mb=0)


def test_settings_add():
    """Test Settings.add() method with batch_rows."""
    # Create base settings
    base_settings = Settings(batch_rows=1000, batch_mb=100)

    # Create settings to add
    add_settings = Settings(batch_rows=2000, batch_mb=200)

    # Add settings
    base_settings.add(add_settings)

    # Verify that values from add_settings override base_settings
    assert base_settings.batch_rows == 2000
    assert base_settings.batch_mb == 200

    # Test with None values (should not override)
    base_settings = Settings(batch_rows=1000, batch_mb=100)
    none_settings = Settings(batch_rows=None, batch_mb=None)

    base_settings.add(none_settings)

    # Verify that None values don't override existing values
    assert base_settings.batch_rows == 1000
    assert base_settings.batch_mb == 100
