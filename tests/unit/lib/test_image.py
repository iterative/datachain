from PIL import Image
from torch import Tensor
from torchvision.transforms import ToTensor

from datachain.lib.file import ImageFile
from datachain.lib.image import (
    convert_image,
    convert_images,
)

IMAGE = Image.new(mode="RGB", size=(64, 64))


def test_convert_image():
    converted_img = convert_image(
        IMAGE,
        mode="RGBA",
        size=(32, 32),
        transform=ToTensor(),
    )
    assert isinstance(converted_img, Tensor)
    assert converted_img.size() == (4, 32, 32)


def test_convert_image_hf(fake_hf_model):
    _, processor = fake_hf_model
    converted_img = convert_image(
        IMAGE,
        transform=processor.image_processor,
    )
    assert isinstance(converted_img, Tensor)


def test_image_file(tmp_path, catalog):
    file_name = "img.jpg"
    file_path = tmp_path / file_name

    IMAGE.save(file_path)

    file = ImageFile(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, caching_enabled=False)
    assert isinstance(file.read(), Image.Image)


def test_convert_images(tmp_path):
    file1_name = "img1.jpg"
    file1_path = tmp_path / file1_name
    file2_name = "img2.jpg"
    file2_path = tmp_path / file2_name

    img1 = Image.new(mode="RGB", size=(64, 64))
    img2 = Image.new(mode="RGB", size=(128, 128))
    img1.save(file1_path)
    img2.save(file2_path)
    images = [img1, img2]

    converted_img = convert_images(
        images,
        mode="RGBA",
        size=(32, 32),
        transform=ToTensor(),
    )
    assert isinstance(converted_img, Tensor)
    assert converted_img.size() == (2, 4, 32, 32)
