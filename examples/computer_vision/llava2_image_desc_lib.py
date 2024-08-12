# pip install accelerate torch huggingface_hub[hf_transfer]
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)

from datachain import C, DataChain, Mapper

# HuggingFace supports the following base models:
#
# "llava-hf/llava-1.5-7b-hf"
# "llava-hf/llava-1.5-13b-hf"
# "llava-hf/bakLlava-v1-hf"
#
# https://huggingface.co/llava-hf
model = "llava-hf/bakLlava-v1-hf"

source = "gs://datachain-demo/dogs-and-cats/"


# device='mps' not supported
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


DEFAULT_FIT_BOX = (500, 500)


def infer_dtype(device):
    if device == "cpu":
        return torch.float32
    return torch.float16


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
            self.model_name, torch_dtype=self.torch_dtype, low_cpu_mem_usage=False
        )
        self.model.to(self.device, self.torch_dtype)

    def process(self, file):
        inputs = self.processor(
            text=self.prompt, images=file.read(), return_tensors="pt"
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
        .filter(C("file.path").glob("*/cat*.jpg"))
        .map(
            desc=LLaVADescribe(
                device=device,
                model=model,
            ),
            params=["file"],
            output={"description": str, "error": str},
        )
        .select("file.source", "file.path", "desc.description", "desc.error")
        .show(2)
    )
