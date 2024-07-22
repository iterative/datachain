# pip install torch
import torch
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    LlavaForConditionalGeneration,
)

from datachain import C, DataChain, Mapper

source = "gs://datachain-demo/dogs-and-cats/"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


DEFAULT_FIT_BOX = (500, 500)


def infer_dtype(device):
    if device == "cpu":
        return torch.float32
    return torch.float16


class BLIP2Describe(Mapper):
    def __init__(self, device="cpu", model="Salesforce/blip2-opt-2.7b", max_tokens=300):
        self.model_name = model
        self.device = device
        self.max_tokens = max_tokens

    def setup(self):
        self.torch_dtype = infer_dtype(self.device)
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=self.torch_dtype
        )
        self.model.to(self.device)

    def process(self, file):
        inputs = self.processor(images=file.read(), return_tensors="pt").to(
            self.device, self.torch_dtype
        )

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        desc = generated_text[0]
        return desc.strip(), ""


class LLaVADescribe(Mapper):
    def __init__(self, device="cpu", model="llava-hf/llava-1.5-7b-hf", max_tokens=300):
        self.device = device
        self.model_name = model
        self.max_tokens = max_tokens
        self.prompt = "USER: <image>\nDescribe this picture\nASSISTANT:"

    def setup(self):
        self.torch_dtype = infer_dtype(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
        )
        self.model.to(self.device)

    def process(self, file):
        inputs = self.processor(
            text=self.prompt, images=file.get_value(), return_tensors="pt"
        ).to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        desc = generated_text[0]
        return desc.split("ASSISTANT:")[-1].strip(), ""


if __name__ == "__main__":
    (
        DataChain.from_storage(source, type="image")
        .filter(C("file.name").glob("cat*.jpg"))
        .map(
            desc=BLIP2Describe(
                # device=device,
                device="cpu",
            ),
            params=["file"],
            output={"description": str, "error": str},
        )
        .select(
            "file.source", "file.parent", "file.name", "desc.description", "desc.error"
        )
        .show(5)
    )
