import open_clip
import pytest
from datasets import load_dataset
from torch import Size, Tensor
from torchvision.datasets import FakeData
from torchvision.transforms import v2

from datachain.lib.dc import DataChain
from datachain.lib.pytorch import PytorchDataset


@pytest.fixture
def fake_image_dir(catalog, tmp_path):
    # Create fake images in labeled dirs
    data_path = tmp_path / "data" / ""
    for i, (img, label) in enumerate(FakeData()):
        label = str(label)
        (data_path / label).mkdir(parents=True, exist_ok=True)
        img.save(data_path / label / f"{i}.jpg")
    return data_path


@pytest.fixture
def fake_dataset(catalog, fake_image_dir):
    # Create dataset from images
    uri = fake_image_dir.as_uri()
    return (
        DataChain.from_storage(uri, type="image")
        .map(text=lambda file: file.parent.split("/")[-1], output=str)
        .map(label=lambda text: int(text), output=int)
        .save("fake")
    )


def test_pytorch_dataset(fake_dataset):
    transform = v2.Compose([v2.ToTensor(), v2.Resize((64, 64))])
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    pt_dataset = PytorchDataset(
        name=fake_dataset.name,
        version=fake_dataset.version,
        transform=transform,
        tokenizer=tokenizer,
    )
    img, text, label = next(iter(pt_dataset))
    assert isinstance(img, Tensor)
    assert isinstance(text, Tensor)
    assert isinstance(label, int)
    assert img.size() == Size([3, 64, 64])


def test_pytorch_dataset_sample(fake_dataset):
    transform = v2.Compose([v2.ToTensor(), v2.Resize((64, 64))])
    pt_dataset = PytorchDataset(
        name=fake_dataset.name,
        version=fake_dataset.version,
        transform=transform,
        num_samples=700,
    )
    assert len(list(pt_dataset)) == 700


def test_to_pytorch(fake_dataset):
    from torch.utils.data import IterableDataset

    transform = v2.Compose([v2.ToTensor(), v2.Resize((64, 64))])
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    pt_dataset = fake_dataset.to_pytorch(transform=transform, tokenizer=tokenizer)
    assert isinstance(pt_dataset, IterableDataset)
    img, text, label = next(iter(pt_dataset))
    assert isinstance(img, Tensor)
    assert isinstance(text, Tensor)
    assert isinstance(label, int)
    assert img.size() == Size([3, 64, 64])


def test_hf_to_pytorch(catalog, fake_image_dir):
    hf_ds = load_dataset("imagefolder", data_dir=fake_image_dir)
    chain = DataChain.from_hf(hf_ds)
    pt_ds = chain.to_pytorch()
    img, label = next(iter(pt_ds))
    assert isinstance(img, Tensor)
    assert label == 0
