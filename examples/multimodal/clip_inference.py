"""
To install the required dependencies:

  uv pip install "datachain[examples]"


Example demonstrates using DataChain Pytorch data loader to stream image
and text object from cloud storage. We pass extracted data to Open CLIP
to compute similarity score between image and its caption.
"""

import open_clip
import torch
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader

import datachain as dc
from datachain import C, func

source = "gs://datachain-demo/50k-laion-files/000000/00000000*"


def create_dataset():
    imgs = dc.read_storage(source, type="image", anon=True).filter(
        C("file.path").glob("*.jpg")
    )
    captions = dc.read_storage(source, type="text", anon=True).filter(
        C("file.path").glob("*.txt")
    )
    return imgs.merge(
        captions,
        on=func.path.file_stem(imgs.c("file.path")),
        right_on=func.path.file_stem(captions.c("file.path")),
    )


if __name__ == "__main__":
    q = create_dataset()

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    ds = q.select("file", "right_file").to_pytorch(
        transform=preprocess,
        tokenizer=tokenizer,
    )
    loader = DataLoader(ds, batch_size=16)

    similarity_sum = 0.0
    row_count = 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for image, text in loader:
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            similarity_sum += (
                cosine_similarity(image_features, text_features).sum().item()
            )
            row_count += len(image_features)

    print("Average cosine similarity:", similarity_sum / row_count)
