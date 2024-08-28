import numpy as np
from datasets import Array2D, Dataset, DatasetDict, Sequence, Value, load_dataset
from datasets.features.image import image_to_bytes
from PIL import Image
from scipy.io.wavfile import write

from datachain.lib.data_model import dict_to_data_model
from datachain.lib.hf import (
    HFClassLabel,
    HFGenerator,
    HFImage,
    get_output_schema,
    stream_splits,
)


def test_hf():
    ds = Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"]})
    schema = get_output_schema(ds)
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
    schema = {"split": str} | get_output_schema(ds_dict["train"])

    gen = HFGenerator(ds_dict, dict_to_data_model("", schema))
    gen.setup()
    row = next(iter(gen.process("train")))

    assert row.split == "train"
    assert row.pokemon == "bulbasaur"


def test_hf_class_label():
    ds = Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"]})
    ds = ds.class_encode_column("pokemon")
    schema = get_output_schema(ds)
    assert schema["pokemon"] is HFClassLabel

    gen = HFGenerator(ds, dict_to_data_model("", schema))
    gen.setup()
    row = next(iter(gen.process()))
    assert row.pokemon.string == "bulbasaur"
    assert row.pokemon.integer == 0


def test_hf_sequence_list():
    ds = Dataset.from_dict({"seq": [[0, 1], [2, 3]]})
    schema = get_output_schema(ds)
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
    schema = get_output_schema(ds)
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
    schema = get_output_schema(ds)
    assert schema["arr"] == list[list[int]]

    gen = HFGenerator(ds, dict_to_data_model("", schema))
    gen.setup()
    row = next(iter(gen.process()))
    assert row.arr == [[0, 1], [2, 3]]


def test_hf_image(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    img = Image.new(mode="RGB", size=(64, 64))
    img.save(train_dir / "img1.png")

    ds = load_dataset("imagefolder", data_dir=tmp_path)
    schema = get_output_schema(ds["train"])
    assert schema["image"] is HFImage

    gen = HFGenerator(ds, dict_to_data_model("", schema))
    gen.setup()
    row = next(iter(gen.process("train")))
    assert row.image.img == image_to_bytes(img)


def test_hf_audio(tmp_path):
    # See https://stackoverflow.com/questions/66191480/how-to-convert-a-numpy-array-to-a-mp3-file
    samplerate = 44100
    fs = 100
    t = np.linspace(0.0, 1.0, samplerate)
    amplitude = np.iinfo(np.int16).max
    data = amplitude * np.sin(2.0 * np.pi * fs * t)
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    write(train_dir / "example.wav", samplerate, data.astype(np.int16))

    ds = load_dataset("audiofolder", data_dir=tmp_path)
    schema = get_output_schema(ds["train"])

    gen = HFGenerator(ds, dict_to_data_model("", schema))
    gen.setup()
    row = next(iter(gen.process("train")))
    assert row.audio.path == str(train_dir / "example.wav")
    assert np.allclose(row.audio.array, data / amplitude, atol=1e-4)
    assert row.audio.sampling_rate == samplerate
