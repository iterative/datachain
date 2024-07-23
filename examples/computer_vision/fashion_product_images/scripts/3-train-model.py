import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from datachain import DataChain
from datachain.lib.pytorch import label_to_int

transform = v2.Compose(
    [
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Resize((64, 64)),
        v2.RGB(),
    ]
)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


#### Train  #####
def train_model(train_loader, num_classes, num_epochs=20, lr=0.001):
    """
    Trains a convolutional neural network on a given dataset.

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        num_classes (int): Number of classes in the dataset.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 20.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        tuple: A tuple containing the trained model and the optimizer used for training.

    """

    model = CNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_loss = []

    for epoch in range(num_epochs):
        for i, data in tqdm(enumerate(train_loader)):  # noqa: B007
            inputs, labels = data
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss.append(loss.item())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        epoch_mean_loss = np.mean(epoch_loss)
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, epoch_mean_loss))
        epoch_loss = []

    print("Finished Training")
    return model, optimizer


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


if __name__ == "__main__":
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
