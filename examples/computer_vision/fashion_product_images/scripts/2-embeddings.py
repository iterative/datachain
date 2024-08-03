import torch
from torchvision import transforms
from torchvision.models import resnet50

from datachain import DataChain
from datachain.lib.image import convert_image

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
    img_raw = file.read()
    img = convert_image(img_raw, transform=transformer).unsqueeze(0)
    with torch.no_grad():
        emb = model(img)

    return emb[0].tolist()


print("\n# Compute and Save Embeddings:")
dc_emb = (
    DataChain(name="fashion-tmp")  # from 2-basic-operations.py
    .limit(1000)
    .map(embeddings=embeddings_processor)
    .save("fashion-embeddings")
)
dc_emb.limit(3)
