import torch
from transformers import CLIPModel, CLIPProcessor

from datachain.lib.file import TextFile
from datachain.lib.text import convert_text


def test_convert_text(fake_clip_model):
    text = "thisismytext"
    model, _, tokenizer = fake_clip_model
    converted_text = convert_text(text, tokenizer=tokenizer)
    assert isinstance(converted_text, torch.Tensor)

    tokenizer_kwargs = {"context_length": 100}
    converted_text = convert_text(
        text, tokenizer=tokenizer, tokenizer_kwargs=tokenizer_kwargs
    )
    assert converted_text.size() == (1, 100)

    converted_text = convert_text(
        text, tokenizer=tokenizer, tokenizer_kwargs=tokenizer_kwargs
    )
    converted_text = convert_text(text, tokenizer=tokenizer, encoder=model.encode_text)
    assert converted_text.dtype == torch.float32


def test_convert_text_hf():
    text = "thisismytext"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    converted_text = convert_text(
        text, tokenizer=processor.tokenizer, encoder=model.get_text_features
    )
    assert converted_text.dtype == torch.float32


def test_text_file_mapper(tmp_path, catalog):
    file_name = "myfile"
    text = "myText"

    file_path = tmp_path / file_name
    with open(file_path, "w") as fd:
        fd.write(text)

    file = TextFile(name=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, caching_enabled=False)
    res = file.get_value()
    assert res == text
