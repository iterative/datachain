"""
To install the required dependencies:

  pip install datachain[examples]

"""

import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)

import datachain as dc
from datachain import C, File

model = "llava-hf/llava-1.5-7b-hf"

# HuggingFace supports the following base models:
#
# "llava-hf/llava-1.5-7b-hf"
# "llava-hf/llava-1.5-13b-hf"
# "llava-hf/bakLlava-v1-hf"
#
# https://huggingface.co/llava-hf


# Probably this code can be written with HF pipeline
# but we keep it a bit more low-level for the sake of example.
class LLaVaProcessor:
    def __init__(self, model_name, max_tokens=300):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.prompt = "USER: <image>\nDescribe this picture\nASSISTANT:"

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=self.dtype, low_cpu_mem_usage=True
        ).to(self.device)


def process(processor: LLaVaProcessor, file: File) -> tuple[str, str]:
    inputs = processor.processor(
        text=processor.prompt, images=file.read(), return_tensors="pt"
    ).to(processor.device, processor.dtype)

    generated_ids = processor.model.generate(
        **inputs, max_new_tokens=processor.max_tokens
    )
    generated_text = processor.processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )
    desc = generated_text[0]
    return desc.split("ASSISTANT:")[-1].strip(), ""


if __name__ == "__main__":
    (
        dc.read_storage("gs://datachain-demo/dogs-and-cats/", type="image", anon=True)
        .filter(C("file.path").glob("*/cat*.jpg"))
        .setup(processor=lambda: LLaVaProcessor(model_name=model))
        .map(process, output=("description", "error"))
        .show(5)
    )
