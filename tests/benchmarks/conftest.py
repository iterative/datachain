from pathlib import Path

import pytest


@pytest.fixture
def bucket():
    return "s3://noaa-bathymetry-pds/"


@pytest.fixture
def datasets():
    return Path(__file__).parent / "datasets"
