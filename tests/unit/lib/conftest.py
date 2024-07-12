import pytest
import torch
from torch import float32
from torchvision.transforms import v2


@pytest.fixture(scope="session")
def fake_clip_model():
    class Model:
        def encode_image(self, tensor):
            return torch.randn(len(tensor), 512)

        def encode_text(self, tensor):
            return torch.randn(len(tensor), 512)

    def tokenizer(tensor, context_length=77):
        return torch.randn(len(tensor), context_length)

    model = Model()
    preprocess = v2.ToDtype(float32, scale=True)
    return model, preprocess, tokenizer
