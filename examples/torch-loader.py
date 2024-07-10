# pip install Pillow torchvision

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from datachain.lib.dc import C, DataChain
from datachain.lib.pytorch import label_to_int

STORAGE = "gs://dvcx-datalakes/dogs-and-cats/"

# Define transformation for data preprocessing
transform = v2.Compose(
    [
        v2.ToTensor(),
        v2.Resize((64, 64)),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


CLASSES = ["cat", "dog"]


# Define torch model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, len(CLASSES))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    ds = (
        DataChain.from_storage(STORAGE, type="image")
        .filter(C.name.glob("*.jpg"))
        .map(label=lambda name: label_to_int(name[:3], CLASSES), output=int)
    )

    train_loader = DataLoader(
        ds.to_pytorch(transform=transform),
        batch_size=16,
        num_workers=2,
    )

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, loss.item()))

    print("Finished Training")
