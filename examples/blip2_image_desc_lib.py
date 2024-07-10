# pip install torch
import torch

from datachain.lib.hf_image_to_text import BLIP2describe
from datachain.query import C, DatasetQuery

source = "gs://dvcx-datalakes/dogs-and-cats/"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
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
        .limit(5)
        .add_signals(
            BLIP2describe(
                # device=device,
                device="cpu",
            ),
            parallel=False,
        )
        .select("source", "parent", "name", "description", "error")
        .results()
    )
    print(*results, sep="\n")
