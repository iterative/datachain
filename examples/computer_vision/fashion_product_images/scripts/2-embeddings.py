import torch

from datachain import DataChain
from datachain.lib.image import convert_image
from torchvision import transforms
from torchvision.models import resnet50


# Model & Transform methods
model = resnet50(pretrained=True).eval()
transformer = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Embeddings processor function

def embeddings_processor(file) -> list[float]: 
    img_raw = file.get_value()
    img = convert_image(img_raw, transform=transformer).unsqueeze(0)
    with torch.no_grad():
        emb = model(img)

    return emb[0].tolist()


# Compute and Save Embeddings

print("\n# Compute and Save Embeddings:")
ds_emb = (
    DataChain(name="fashion-test")
    .limit(1000)
    .map(embeddings=embeddings_processor)
    .save("fashion-embeddings")
)
ds_emb.limit(3)
