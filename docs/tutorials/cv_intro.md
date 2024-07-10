# Get started for computer vision

Learn how to use DataChain to:
- Generate a dataset from cloud storage or local directory
- Apply transformations to add metadata to a dataset
- Consume that dataset into PyTorch data loaders for model training

## Creating datasets from files

We will start with a collection of cat and dog images stored in s3:

```python
from datachain.lib.dc import DataChain

ds = DataChain("s3://dvc-public/data/dvcx/cats-and-dogs/train/*.jpg")
```

DataChain works lazily so as not to waste compute. To force it to run, let's convert it to a pandas dataframe:

```
ds.to_pandas()
```

DataChain will scan for matching files and create a dataset from them. The output looks like:

```
      id vtype  dir_type         parent          name  ...
0      3               0  dogs-and-cats     cat.1.jpg  ...
1      5               0  dogs-and-cats    cat.10.jpg  ...
2      7               0  dogs-and-cats   cat.100.jpg  ...
3      9               0  dogs-and-cats  cat.1000.jpg  ...
4     11               0  dogs-and-cats  cat.1001.jpg  ...
..   ...   ...       ...            ...           ...
195  393               0  dogs-and-cats  dog.1084.jpg  ...
196  395               0  dogs-and-cats  dog.1085.jpg  ...
197  397               0  dogs-and-cats  dog.1086.jpg  ...
198  399               0  dogs-and-cats  dog.1087.jpg  ...
199  401               0  dogs-and-cats  dog.1088.jpg  ...
```

DataChain automatically captures these file attributes for each file. A collection of columns is called a feature, and these columns are all part of the `File` feature. We will take a look
below at how to work with these columns and the files themselves.

## Map new feature onto the dataset

Let's add some labels to our dataset so we can train on it. Use `ds.map()` to
add new features to the dataset:

```python
ds = ds.map(lambda name: (name[:3],), output={"label": str})
```

The first argument can be any Python function (we call this a user-defined function or
UDF) to apply to each row in the dataset. By using `name` as the input to the function,
DataChain knows to pass the value from the `name` column to the function (or you can use the
`params` argument to pass the column names explicitly). The first 3 letters of each
filename in this case represent the label (cat or dog). The UDF must return a tuple of
values, each corresponding to a column/feature in the output.

The second argument defines the column names and types for the output. In this case, it
adds a single `label` column with string values.

Let's check the output again:

```
ds.select("name", "label").to_pandas()
```

Now the output looks like:

```
             name label
0       cat.1.jpg   cat
1      cat.10.jpg   cat
2     cat.100.jpg   cat
3    cat.1000.jpg   cat
4    cat.1001.jpg   cat
..            ...   ...
195  dog.1084.jpg   dog
196  dog.1085.jpg   dog
197  dog.1086.jpg   dog
198  dog.1087.jpg   dog
199  dog.1088.jpg   dog
```

## Model training

Getting a dataframe is great, but for model training, we need to:
- Read the images themselves
- Convert those images to vectors and apply other transforms to them
- Convert the labels from strings to encoded integers (like cat=0, dog=1)
- In a more realistic scenario, iterate over a large collection without killing performance or exploding memory

`ds.to_pytorch()` creates a PyTorch
[IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)
that can be passed to the standard PyTorch
[DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), and
it can even apply `torchvision`
[transforms](https://pytorch.org/vision/stable/transforms.html).

Either features or feature reader objects can be passed to `ds.to_pytorch()`. Each
reader can transform the feature values as needed:

```python
from datachain.lib.image import ImageReader
from datachain.lib.reader import LabelReader
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# Define transformation for data preprocessing
transform = v2.Compose(
    [
        v2.ToTensor(),
        v2.Resize((64, 64)),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Create a pytorch dataset
pytorch_ds = ds.to_pytorch(
    ImageReader(),
    LabelReader(feature="label", classes=["cat", "dog"]),
    transform=transform,
)

# Pass to standard pytorch dataloader
train_loader = DataLoader(
    pytorch_dataset,
    batch_size=16,
)

# Train the model
train(train_loader)
```

<details>
<summary>
Get example model code
</summary>

To run this example, you can use this simple Pytorch model code:

```python
import torch
from torch import nn, optim


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
        x = self.fc2(x)
        return x


# Define training loop
def train(train_loader):
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
```

</details>

The only DataChain code in the above block is here:

```python
# Create a pytorch dataset
pytorch_ds = ds.to_pytorch(
    ImageReader(),
    LabelReader(feature="label", classes=["cat", "dog"]),
    transform=transform,
)
```

Let's take a closer look at what this code does. It's reading and returning pairs of
values for two different features:
- `ImageReader()` reads in each file and returns it as a [PIL
  Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image)
  that can be passed to PyTorch
- `LabelReader(feature="label", classes=["cat", "dog"])` reads the `label` feature
  and converts it to integer-encoded labels for the classes cat and dog

`ds.to_pytorch()` will wrap these values into a PyTorch dataset, optimize streaming the
data to PyTorch, and apply any transforms so you don't have to wrangle your data into a
special format for training.
