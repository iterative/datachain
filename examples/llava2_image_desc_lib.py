# pip install torch
import torch

from datachain.lib.hf_image_to_text import LLaVAdescribe
from datachain.query import C, DatasetQuery

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
    results = (
        DatasetQuery(
            source,
            anon=True,
        )
        .filter(C.name.glob("cat*.jpg"))
        .limit(2)
        .add_signals(
            LLaVAdescribe(
                device=device,
                model=model,
            ),
            parallel=False,
        )
        .select("source", "parent", "name", "description", "error")
        .results()
    )
    print(*results, sep="\n")
