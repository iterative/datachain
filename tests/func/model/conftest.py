import os

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def running_img() -> np.ndarray:
    img_file = os.path.join(os.path.dirname(__file__), "data", "running.jpg")
    return np.array(Image.open(img_file))


@pytest.fixture
def ships_img() -> np.ndarray:
    img_file = os.path.join(os.path.dirname(__file__), "data", "ships.jpg")
    return np.array(Image.open(img_file))


@pytest.fixture
def running_img_masks() -> torch.Tensor:
    mask0_file = os.path.join(os.path.dirname(__file__), "data", "running-mask0.png")
    mask0_np = np.array(Image.open(mask0_file))

    mask1_file = os.path.join(os.path.dirname(__file__), "data", "running-mask1.png")
    mask1_np = np.array(Image.open(mask1_file))

    return torch.tensor([mask0_np.astype(np.float32), mask1_np.astype(np.float32)])
