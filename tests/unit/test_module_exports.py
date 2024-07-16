# flake8: noqa: F401

import builtins
import sys

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
            DataModel,
            File,
            FileBasic,
            FileError,
            Generator,
            ImageFile,
            IndexedFile,
            Mapper,
            Session,
            TarVFile,
            TextFile,
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


@pytest.mark.parametrize("dep", ["torch", "torchvision", "transformers"])
def test_no_torch_deps(monkeypatch, dep):
    real_import = builtins.__import__

    def monkey_import_importerror(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name.startswith(dep):
            raise ImportError(f"Mocked import error {name}")
        return real_import(
            name, globals=globals, locals=locals, fromlist=fromlist, level=level
        )

    for module in list(sys.modules):
        if module.startswith((dep, "datachain")):
            monkeypatch.delitem(sys.modules, module)
    monkeypatch.setattr(builtins, "__import__", monkey_import_importerror)

    try:
        from datachain import (
            AbstractUDF,
            Aggregator,
            BaseUDF,
            C,
            Column,
            DataChain,
            DataChainError,
            DataModel,
            File,
            FileBasic,
            FileError,
            Generator,
            ImageFile,
            IndexedFile,
            Mapper,
            Session,
            TarVFile,
            TextFile,
        )
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Importing raised an exception: {e}")

    with pytest.raises(ImportError):
        from datachain.torch import (
            PytorchDataset,
            clip_similarity_scores,
            convert_image,
            convert_images,
            convert_text,
            label_to_int,
        )
