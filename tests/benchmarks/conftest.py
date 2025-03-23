from pathlib import Path

import pytest


@pytest.fixture
def datasets():
    return Path(__file__).parent / "datasets"
