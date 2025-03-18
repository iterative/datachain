import pytest

from datachain import DataChain
from datachain.error import DatasetNotFoundError
from datachain.query.session import Session


def test_listing_dataset_lifecycle(tmp_path, catalog):
    p = tmp_path / "hello.txt"
    p.write_text("hello", encoding="utf-8")

    session_name = "asd3d5"
    ds_name = "my_test_ds13"

    with pytest.raises(ValueError):
        with Session(session_name, catalog=catalog):
            ds_name = "my_test_ds13"
            DataChain.from_storage(str(tmp_path)).exec()
            DataChain.from_values(key=["a", "b", "c"]).save(ds_name)
            raise ValueError("This is a test exception")

    with pytest.raises(DatasetNotFoundError):
        tmp_path, catalog.get_dataset(ds_name)

    assert DataChain.listings(catalog=catalog).count() == 1
