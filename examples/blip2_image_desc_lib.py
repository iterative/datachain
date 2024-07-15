# pip install torch
import torch

from datachain.lib.dc import C, DataChain
from datachain.lib.hf_image_to_text import BLIP2Describe

source = "gs://dvcx-datalakes/dogs-and-cats/"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


if __name__ == "__main__":
    (
        DataChain.from_storage(source, type="image")
        .filter(C.name.glob("cat*.jpg"))
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
