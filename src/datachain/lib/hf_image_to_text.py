try:
    import numpy as np
    import torch
    from PIL import Image, ImageOps, UnidentifiedImageError
    from transformers import (
        AutoProcessor,
        Blip2ForConditionalGeneration,
        Blip2Processor,
        LlavaForConditionalGeneration,
    )
except ImportError as exc:
    raise ImportError(
        "Missing dependencies for computer vision:\n"
        "To install run:\n\n"
        "  pip install 'datachain[cv]'\n"
    ) from exc


from datachain.query import Object, udf
from datachain.sql.types import String

DEFAULT_FIT_BOX = (500, 500)


def encode_image(raw):
    try:
        img = Image.open(raw)
    except UnidentifiedImageError:
        return None
    img.load()
    img = img.convert("RGB")
    return ImageOps.fit(img, DEFAULT_FIT_BOX)


def infer_dtype(device):
    if device == "cpu":
        return torch.float32
    return torch.float16


@udf(
    params=(Object(encode_image),),  # Columns consumed by the UDF.
    output={
        "description": String,
        "error": String,
    },  # Signals being returned by the UDF.
    batch=64,
    method="describe",
)
class BLIP2describe:
    def __init__(self, device="cpu", model="Salesforce/blip2-opt-2.7b", max_tokens=300):
        self.torch_dtype = infer_dtype(device)
        self.processor = Blip2Processor.from_pretrained(model)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model, torch_dtype=self.torch_dtype
        )
        self.device = device
        self.model.to(device)
        self.max_tokens = max_tokens

    def describe(self, imgs):
        images = np.squeeze(np.asarray(imgs))
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.device, self.torch_dtype
        )

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [(desc.strip(), "") for desc in generated_text]


@udf(
    params=(Object(encode_image),),  # Columns consumed by the UDF.
    output={
        "description": String,
        "error": String,
    },  # Signals being returned by the UDF.
    batch=16,
    method="describe",
)
class LLaVAdescribe:
    def __init__(self, device="cpu", model="llava-hf/llava-1.5-7b-hf", max_tokens=300):
        self.device = device
        self.torch_dtype = infer_dtype(device)
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
        )
        self.model.to(device)
        self.max_tokens = max_tokens
        self.prompt = "USER: <image>\nDescribe this picture\nASSISTANT:"

    def describe(self, imgs):
        images = np.squeeze(np.asarray(imgs))
        inputs = self.processor(
            text=[self.prompt] * len(imgs), images=images, return_tensors="pt"
        ).to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [(desc.split("ASSISTANT:")[-1].strip(), "") for desc in generated_text]
