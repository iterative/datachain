import open_clip
import torch
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader

from datachain import C, DataChain

source = "gs://datachain-demo/50k-laion-files/000000/00000000*"


def create_dataset():
    imgs = (
        DataChain.from_storage(source, type="image")
        .filter(C("file.path").glob("*.jpg"))
        .map(stem=lambda file: file.get_file_stem(), params=["file"], output=str)
    )
    captions = (
        DataChain.from_storage(source, type="text")
        .filter(C("file.path").glob("*.txt"))
        .map(stem=lambda file: file.get_file_stem(), params=["file"], output=str)
    )
    return imgs.merge(captions, on="stem")


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

    similarity_sum = 0
    row_count = 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for image, text in loader:
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            similarity_sum += (
                cosine_similarity(image_features, text_features).sum().item()
            )
            row_count += len(image_features)

    print("Average cosine similarity:", similarity_sum / row_count)
