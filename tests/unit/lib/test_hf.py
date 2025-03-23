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

    gen = HFGenerator(ds, dict_to_data_model("", schema))
    gen.setup()
    row = next(iter(gen.process()))
    assert row.pokemon == "bulbasaur"


def test_hf_split():
    ds_train = Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"]})
    ds_test = Dataset.from_dict({"pokemon": ["charizard", "pikachu"]})
    ds_dict = DatasetDict({"train": ds_train, "test": ds_test})
    ds_dict = stream_splits(ds_dict)
    schema = {"split": str} | get_output_schema(ds_dict["train"].features)

    gen = HFGenerator(ds_dict, dict_to_data_model("", schema))
    gen.setup()
    row = next(iter(gen.process("train")))

    assert row.split == "train"
    assert row.pokemon == "bulbasaur"


def test_hf_class_label():
    ds = Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"]})
    ds = ds.class_encode_column("pokemon")
    schema = get_output_schema(ds.features)
    assert schema["pokemon"] is HFClassLabel

    gen = HFGenerator(ds, dict_to_data_model("", schema))
    gen.setup()
    row = next(iter(gen.process()))
    assert row.pokemon.string == "bulbasaur"
    assert row.pokemon.integer == 0


def test_hf_sequence_list():
    ds = Dataset.from_dict({"seq": [[0, 1], [2, 3]]})
    schema = get_output_schema(ds.features)
    assert schema["seq"] == list[int]

    gen = HFGenerator(ds, dict_to_data_model("", schema))
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

    gen = HFGenerator(ds, dict_to_data_model("", schema))
    gen.setup()
    row = next(iter(gen.process()))
    assert row.pokemon.name == ["bulbasaur"]


def test_hf_array():
    ds = Dataset.from_dict({"arr": [[[0, 1], [2, 3]]]})
    new_features = ds.features.copy()
    new_features["arr"] = Array2D(shape=(2, 2), dtype="int32")
    ds = ds.cast(new_features)
    schema = get_output_schema(ds.features)
    assert schema["arr"] == list[list[int]]

    gen = HFGenerator(ds, dict_to_data_model("", schema))
    gen.setup()
    row = next(iter(gen.process()))
    assert row.arr == [[0, 1], [2, 3]]
