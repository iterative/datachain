from typing import Callable, Optional, Union

import torch
from PIL import Image


def convert_image(
    img: Image.Image,
    mode: str = "RGB",
    size: Optional[tuple[int, int]] = None,
    transform: Optional[Callable] = None,
    encoder: Optional[Callable] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[Image.Image, torch.Tensor]:
    """
    Resize, transform, and otherwise convert an image.

    Args:
        img (Image): PIL.Image object.
        mode (str): PIL.Image mode.
        size (tuple[int, int]): Size in (width, height) pixels for resizing.
        transform (Callable): Torchvision transform or huggingface processor to apply.
        encoder (Callable): Encode image using model.
        device (str or torch.device): Device to use.
    """
    if mode:
        img = img.convert(mode)
    if size:
        img = img.resize(size)
    if transform:
        img = transform(img)

        try:
            from transformers.image_processing_utils import BaseImageProcessor

            if isinstance(transform, BaseImageProcessor):
                img = torch.tensor(img.pixel_values[0])  # type: ignore[assignment,attr-defined]
        except ImportError:
            pass
        if device:
            img = img.to(device)  # type: ignore[attr-defined]
        if encoder:
            img = img.unsqueeze(0)  # type: ignore[attr-defined]
    if encoder:
        img = encoder(img)
    return img


def convert_images(
    images: Union[Image.Image, list[Image.Image]],
    mode: str = "RGB",
    size: Optional[tuple[int, int]] = None,
    transform: Optional[Callable] = None,
    encoder: Optional[Callable] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[list[Image.Image], torch.Tensor]:
    """
    Resize, transform, and otherwise convert one or more images.

    Args:
        images (Image, list[Image]): PIL.Image object or list of objects.
        mode (str): PIL.Image mode.
        size (tuple[int, int]): Size in (width, height) pixels for resizing.
        transform (Callable): Torchvision transform or huggingface processor to apply.
        encoder (Callable): Encode image using model.
        device (str or torch.device): Device to use.
    """
    if isinstance(images, Image.Image):
        images = [images]

    converted = [
        convert_image(img, mode, size, transform, device=device) for img in images
    ]

    if isinstance(converted[0], torch.Tensor):
        converted = torch.stack(converted)  # type: ignore[assignment,arg-type]

    if encoder:
        converted = encoder(converted)

    return converted  # type: ignore[return-value]
