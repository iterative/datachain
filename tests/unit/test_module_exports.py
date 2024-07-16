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
            Feature,
            File,
            FileError,
            FileFeature,
            Generator,
            ImageFile,
            IndexedFile,
            Mapper,
            Session,
            TarVFile,
            TextFile,
            pydantic_to_feature,
        )
        from datachain.torch import (
            PytorchDataset,
            clip_similarity_scores,
            convert_image,
            convert_images,
            convert_text,
            label_to_int,
        )
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Importing raised an exception: {e}")
