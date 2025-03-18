import numpy as np
from datasets import load_dataset
from datasets.features.image import image_to_bytes
from PIL import Image
from scipy.io.wavfile import write

from datachain.lib.data_model import dict_to_data_model
from datachain.lib.hf import (
    HFGenerator,
    HFImage,
    get_output_schema,
)


def test_hf_image(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    img = Image.new(mode="RGB", size=(64, 64))
    img.save(train_dir / "img1.png")

    ds = load_dataset("imagefolder", data_dir=tmp_path)
    schema = {"split": str} | get_output_schema(ds["train"].features)
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
    schema = {"split": str} | get_output_schema(ds["train"].features)

    gen = HFGenerator(ds, dict_to_data_model("", schema))
    gen.setup()
    row = next(iter(gen.process("train")))
    assert row.audio.path == str(train_dir / "example.wav")
    assert np.allclose(row.audio.array, data / amplitude, atol=1e-4)
    assert row.audio.sampling_rate == samplerate
