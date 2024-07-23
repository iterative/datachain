from torch.utils.data import DataLoader

from datachain import DataChain
from datachain.lib.pytorch import label_to_int

# For Local Development
from src.train import train_model, transform

# For Studio
# from train import train_model, transform


CLASSES = [
    "Casual",
    "Ethnic",
    "Sports",
    "Formal",
    "Party",
    "Smart Casual",
    "Travel",
    "nan",
]
NUM_CLASSES = len(CLASSES)


def add_target_label(usage) -> str:
    return usage if usage in CLASSES else "nan"


dc = (
    DataChain(name="fashion-train")
    .map(target=add_target_label, params=["usage"], output=str)
    .map(label=lambda target: label_to_int(target, CLASSES), output=int)
    .limit(1000)  # Take a sample for the DEMO purposes
    .shuffle()
)

print(dc.to_pandas().target.value_counts())

train_loader = DataLoader(
    dc.select("file", "label").to_pytorch(transform=transform),
    batch_size=2,
    num_workers=1,
)

model, optimizer = train_model(train_loader, NUM_CLASSES, num_epochs=3, lr=0.001)

dc.select("file.name", "target", "label").show(3)

# NOTE: Studio requires the  Last line to be a DataChain instance
dc.limit(3)
