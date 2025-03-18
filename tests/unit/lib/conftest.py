import random

import pytest
import torch
from torchvision.transforms import v2
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

TO_TENSOR = v2.Compose(
    [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((64, 64))]
)


@pytest.fixture()
def fake_clip_model():
    class Model(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            self._parameters = {"p_1": torch.nn.Parameter(torch.tensor(1.0))}

        def encode_image(self, tensor):
            return torch.randn(len(tensor), 512)

        def encode_text(self, tensor):
            return torch.randn(len(tensor), 512)

    def tokenizer(text, context_length=77):
        return torch.randn(len(text), context_length)

    model = Model()
    preprocess = TO_TENSOR
    return model, preprocess, tokenizer


@pytest.fixture()
def fake_hf_model():
    class Model(PreTrainedModel):
        def __init__(self, *args, **kwargs):
            self._parameters = {"p_1": torch.nn.Parameter(torch.tensor(1.0))}

        def get_text_features(self, tensor):
            return torch.randn(len(tensor), 512)

        def get_image_features(self, tensor):
            return torch.randn(len(tensor), 512)

    class Tokenizer(PreTrainedTokenizer):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, text):
            input_ids = [
                [random.randint(0, 1000) for _ in range(77)]  # noqa: S311
                for _ in range(len(text))
            ]
            return BatchFeature({"input_ids": input_ids})

        def __bool__(self):
            return True

    class ImageProcessor(BaseImageProcessor):
        def __call__(self, images):
            if not isinstance(images, list):
                images = [images]
            pixel_values = [TO_TENSOR(img) for img in images]
            return BatchFeature({"pixel_values": pixel_values})

    class Processor:
        tokenizer = Tokenizer()
        image_processor = ImageProcessor()

    model = Model()
    processor = Processor()
    return model, processor
