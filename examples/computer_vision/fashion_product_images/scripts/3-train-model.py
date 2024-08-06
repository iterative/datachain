from torch.utils.data import DataLoader

from datachain import DataChain
from datachain.lib.pytorch import label_to_int
from src.train import train_model, transform

# Define classes

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


# Create a Target column


def add_target_label(usage) -> str:
    return usage if usage in CLASSES else "nan"


ds = (
    DataChain(name="fashion-train")
    .map(target=add_target_label, params=["usage"], output=str)
    .map(label=lambda target: label_to_int(target, CLASSES), output=int)
    .limit(1000)  # Take a sample for the DEMO purposes
    .shuffle()
)

print(ds.to_pandas().target.value_counts())

# PyTorch DataLoader

train_loader = DataLoader(
    ds.select("file", "label").to_pytorch(transform=transform),
    batch_size=2,
    num_workers=1,
)

# Train the model

model, optimizer = train_model(train_loader, NUM_CLASSES, num_epochs=3, lr=0.001)

# NOTE: DataChain requires the  Last line to be an instance of DatasetQuery
ds.select("file.name", "target", "label").show(3)
