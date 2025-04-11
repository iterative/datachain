import orjson
import pytest
from sqlalchemy.dialects import sqlite

from datachain.sql.types import JSON  # Corrected import


def test_json_type_process_bind_param():
    """Test JSON process_bind_param with dict and None values."""
    json_type = JSON()  # Corrected instantiation
    dialect = sqlite.dialect()

    # Test with a dictionary value (covers line 343)
    dict_value = {"a": 1, "b": "test"}
    expected_json_string = orjson.dumps(dict_value).decode("utf-8")
    assert json_type.process_bind_param(dict_value, dialect) == expected_json_string

    # Test with None value (covers line 344)
    assert json_type.process_bind_param(None, dialect) is None

    # Test with an empty dictionary
    empty_dict_value = {}
    expected_empty_json_string = orjson.dumps(empty_dict_value).decode("utf-8")
    assert (
        json_type.process_bind_param(empty_dict_value, dialect)
        == expected_empty_json_string
    )


def test_json_type_on_read_convert():
    """Test JSON on_read_convert with JSON string and None."""
    json_type = JSON()
    dialect = sqlite.dialect()

    # Test with a JSON string value
    json_string = '{"a": 1, "b": "test"}'
    expected_dict = {"a": 1, "b": "test"}
    # Use on_read_convert instead of process_result_value
    assert json_type.on_read_convert(json_string, dialect) == expected_dict

    # Test with None value
    assert json_type.on_read_convert(None, dialect) is None

    # Test with an empty JSON object string
    empty_json_string = "{}"
    expected_empty_dict = {}
    assert (
        json_type.on_read_convert(empty_json_string, dialect) == expected_empty_dict
    )
