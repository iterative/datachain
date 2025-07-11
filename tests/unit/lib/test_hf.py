from datasets import Array2D, Dataset, DatasetDict, Sequence, Value

from datachain.lib.data_model import dict_to_data_model
from datachain.lib.hf import (
    HFClassLabel,
    HFGenerator,
    get_output_schema,
    stream_splits,
)


def test_hf():
    ds = Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"]})
    schema = get_output_schema(ds.features)
    assert schema["pokemon"] is str

    gen = HFGenerator(
        ds, dict_to_data_model("", schema, original_names=list(ds.features.keys()))
    )
    gen.setup()
    row = next(iter(gen.process()))
    assert row.pokemon == "bulbasaur"


def test_hf_split():
    ds_train = Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"]})
    ds_test = Dataset.from_dict({"pokemon": ["charizard", "pikachu"]})
    ds_dict = DatasetDict({"train": ds_train, "test": ds_test})
    ds_dict = stream_splits(ds_dict)
    schema = {"split": str} | get_output_schema(ds_dict["train"].features)

    gen = HFGenerator(
        ds_dict,
        dict_to_data_model(
            "",
            schema,
            original_names=["split"] + list(ds_dict["train"].features.keys()),
        ),
    )
    gen.setup()
    row = next(iter(gen.process("train")))

    assert row.split == "train"
    assert row.pokemon == "bulbasaur"


def test_hf_class_label():
    ds = Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"]})
    ds = ds.class_encode_column("pokemon")
    schema = get_output_schema(ds.features)
    assert schema["pokemon"] is HFClassLabel

    gen = HFGenerator(
        ds, dict_to_data_model("", schema, original_names=list(ds.features.keys()))
    )
    gen.setup()
    row = next(iter(gen.process()))
    assert row.pokemon.string == "bulbasaur"
    assert row.pokemon.integer == 0


def test_hf_sequence_list():
    ds = Dataset.from_dict({"seq": [[0, 1], [2, 3]]})
    schema = get_output_schema(ds.features)
    assert schema["seq"] == list[int]

    gen = HFGenerator(
        ds, dict_to_data_model("", schema, original_names=list(ds.features.keys()))
    )
    gen.setup()
    row = next(iter(gen.process()))
    assert row.seq == [0, 1]


def test_hf_sequence_dict():
    ds = Dataset.from_dict(
        {"pokemon": [{"name": ["bulbasaur"]}, {"name": ["squirtle"]}]}
    )
    new_features = ds.features.copy()
    new_features["pokemon"] = Sequence(feature={"name": Value(dtype="string")})
    ds = ds.cast(new_features)
    schema = get_output_schema(ds.features)
    assert schema["pokemon"].model_fields["name"].annotation == list[str]

    gen = HFGenerator(
        ds, dict_to_data_model("", schema, original_names=list(ds.features.keys()))
    )
    gen.setup()
    row = next(iter(gen.process()))
    assert row.pokemon.name == ["bulbasaur"]


def test_hf_sequence_dict_with_invalid_names():
    """Test nested dictionary features with invalid Python identifier names."""
    ds = Dataset.from_dict(
        {"pokemon": [{"name?": ["bulbasaur"]}, {"name?": ["squirtle"]}]}
    )
    new_features = ds.features.copy()
    new_features["pokemon"] = Sequence(feature={"name?": Value(dtype="string")})
    ds = ds.cast(new_features)
    schema = get_output_schema(ds.features)

    # The nested field should keep the original name
    nested_model = schema["pokemon"]
    assert "name?" in nested_model.model_fields

    gen = HFGenerator(
        ds, dict_to_data_model("", schema, original_names=list(ds.features.keys()))
    )
    gen.setup()
    row = next(iter(gen.process()))

    # Should be able to access the nested field with original name
    nested_obj = row.pokemon
    nested_dict = nested_obj.model_dump()
    assert nested_dict["name?"] == ["bulbasaur"]


def test_hf_array():
    ds = Dataset.from_dict({"arr": [[[0, 1], [2, 3]]]})
    new_features = ds.features.copy()
    new_features["arr"] = Array2D(shape=(2, 2), dtype="int32")
    ds = ds.cast(new_features)
    schema = get_output_schema(ds.features)
    assert schema["arr"] == list[list[int]]

    gen = HFGenerator(
        ds, dict_to_data_model("", schema, original_names=list(ds.features.keys()))
    )
    gen.setup()
    row = next(iter(gen.process()))
    assert row.arr == [[0, 1], [2, 3]]


def test_hf_invalid_column_names():
    """Test that datasets with invalid Python identifier column names work correctly."""
    # Create a dataset with column names that aren't valid Python identifiers
    ds = Dataset.from_dict(
        {
            "factual?": ["yes", "no"],
            "user-name": ["alice", "bob"],
            "123column": ["value1", "value2"],
            "valid_name": ["data1", "data2"],
        }
    )

    schema = get_output_schema(ds.features)

    # Test with original column names preserved
    gen = HFGenerator(
        ds, dict_to_data_model("", schema, original_names=list(ds.features.keys()))
    )
    gen.setup()
    row = next(iter(gen.process()))

    # The field names should remain as the original names
    assert hasattr(row, 'factual?')
    assert hasattr(row, 'user-name')
    assert hasattr(row, '123column')
    assert hasattr(row, 'valid_name')

    # Verify the data is correct
    assert getattr(row, 'factual?') == "yes"
    assert getattr(row, 'user-name') == "alice"
    assert getattr(row, '123column') == "value1"
    assert row.valid_name == "data1"


def test_hf_invalid_column_names_with_read_hf():
    """Test that read_hf handles invalid Python identifier column names correctly."""
    # Create a dataset with column names that aren't valid Python identifiers
    ds = Dataset.from_dict(
        {
            "factual?": ["yes", "no"],
            "user-name": ["alice", "bob"],
            "123column": ["value1", "value2"],
            "valid_name": ["data1", "data2"],
        }
    )

    # This should not raise an error with the fix
    # Mock the read_values function to avoid needing a full session
    import unittest.mock

    from datachain.lib.dc.hf import read_hf

    with unittest.mock.patch("datachain.lib.dc.values.read_values") as mock_read_values:
        mock_chain = unittest.mock.Mock()
        mock_read_values.return_value = mock_chain

        # This should work without raising KeyError
        result = read_hf(ds)

        # Verify that read_values was called
        mock_read_values.assert_called_once()

        # Verify that gen was called on the chain
        mock_chain.gen.assert_called_once()

        # Get the HFGenerator and output schema that were passed to gen
        gen_call = mock_chain.gen.call_args
        hf_generator = gen_call[0][0]  # First positional argument
        output_schema = gen_call[1]['output']  # output keyword argument

        # The output schema should contain the original column names
        assert 'factual?' in output_schema
        assert 'user-name' in output_schema
        assert '123column' in output_schema
        assert 'valid_name' in output_schema
