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

    # The nested field should be accessible with normalized name
    nested_model = schema["pokemon"]
    assert "name_" in nested_model.model_fields or "name?" in nested_model.model_fields

    gen = HFGenerator(
        ds, dict_to_data_model("", schema, original_names=list(ds.features.keys()))
    )
    gen.setup()
    row = next(iter(gen.process()))

    # Should be able to access the nested field regardless of normalization
    nested_obj = row.pokemon
    nested_dict = nested_obj.model_dump()
    assert "bulbasaur" in str(nested_dict)  # The value should be present


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

    # Check that we can access the data using the normalized field names
    # The exact field names will depend on how normalize_col_names transforms them
    assert hasattr(row, "factual_")  # ? replaced with _
    assert hasattr(row, "user_name")  # - replaced with _
    assert hasattr(row, "c0_123column")  # starts with number, gets prefix
    assert hasattr(row, "valid_name")  # already valid, unchanged

    # Verify the data is correct
    assert row.factual_ == "yes"
    assert row.user_name == "alice"
    assert row.c0_123column == "value1"
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

    with unittest.mock.patch("datachain.lib.dc.hf.read_values") as mock_read_values:
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
        hf_generator, output_schema = gen_call[0]

        # The output schema should contain the original column names
        assert (
            "factual?" in output_schema or "factual_" in output_schema
        )  # May be normalized
        assert (
            "user-name" in output_schema or "user_name" in output_schema
        )  # May be normalized
        assert (
            "123column" in output_schema or "c0_123column" in output_schema
        )  # May be normalized
        assert "valid_name" in output_schema
