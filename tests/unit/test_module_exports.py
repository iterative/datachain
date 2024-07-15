# flake8: noqa: F401

import pytest


def test_module_exports():
    try:
        from datachain import (
            AbstractUDF,
            Aggregator,
            BaseUDF,
            C,
            Column,
            DataChain,
            DataChainError,
            File,
            FileError,
            FileFeature,
            Generator,
            IndexedFile,
            Mapper,
            Session,
            TarVFile,
            VersionedModel,
            convert_images,
            convert_text,
        )
        from datachain.image import ImageFile, convert_images
        from datachain.text import convert_text
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Importing raised an exception: {e}")
