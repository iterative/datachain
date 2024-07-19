# pip install accelerate torch
import torch

from datachain.lib.dc import C, DataChain
from datachain.lib.hf_image_to_text import LLaVADescribe

model = "llava-hf/llava-1.5-7b-hf"

# HuggingFace supports the following base models:
#
# "llava-hf/llava-1.5-7b-hf"
# "llava-hf/llava-1.5-13b-hf"
# "llava-hf/bakLlava-v1-hf"
#
# https://huggingface.co/llava-hf

source = "gs://dvcx-datalakes/dogs-and-cats/"

# device='mps' not supported
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__ == "__main__":
    (
        DataChain.from_storage(source, type="image")
        .filter(C.name.glob("cat*.jpg"))
        .map(
            desc=LLaVADescribe(
                device=device,
                model=model,
            ),
            params=["file"],
            output={"description": str, "error": str},
        )
        .select(
            "file.source", "file.parent", "file.name", "desc.description", "desc.error"
        )
        .show(2)
    )
