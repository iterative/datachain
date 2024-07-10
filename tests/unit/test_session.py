import re

import pytest
import sqlalchemy as sa

from datachain.error import DatasetNotFoundError
from datachain.query import DatasetQuery, Session
from datachain.sql.types import String


def test_ephemeral_dataset_naming(catalog):
    session_name = "qwer45"

    with pytest.raises(ValueError):
        Session("wrong-ds_name", catalog=catalog)

    with Session(session_name, catalog=catalog) as session:
        ds_name = "my_test_ds12"
        session.catalog.create_dataset(ds_name, columns=(sa.Column("name", String),))
        ds_tmp = DatasetQuery(
            name=ds_name, session=session, catalog=session.catalog
        ).save()
        session_uuid = f"[0-9a-fA-F]{{{Session.SESSION_UUID_LEN}}}"
        table_uuid = f"[0-9a-fA-F]{{{Session.TEMP_TABLE_UUID_LEN}}}"

        name_prefix = f"{Session.DATASET_PREFIX}{session_name}"
        pattern = rf"^{name_prefix}_{session_uuid}_{table_uuid}$"

        assert re.match(pattern, ds_tmp.name) is not None


def test_global_session_naming(catalog):
    session_uuid = f"[0-9a-fA-F]{{{Session.SESSION_UUID_LEN}}}"
    table_uuid = f"[0-9a-fA-F]{{{Session.TEMP_TABLE_UUID_LEN}}}"

    ds_name = "qwsd"
    catalog.create_dataset(ds_name, columns=(sa.Column("name", String),))
    ds_tmp = DatasetQuery(name=ds_name, catalog=catalog).save()
    global_prefix = f"{Session.DATASET_PREFIX}{Session.GLOBAL_SESSION_NAME}"
    pattern = rf"^{global_prefix}_{session_uuid}_{table_uuid}$"
    assert re.match(pattern, ds_tmp.name) is not None


def test_ephemeral_dataset_lifecycle(catalog):
    session_name = "asd3d4"
    with Session(session_name, catalog=catalog) as session:
        ds_name = "my_test_ds12"
        session.catalog.create_dataset(ds_name, columns=(sa.Column("name", String),))
        ds_tmp = DatasetQuery(
            name=ds_name, session=session, catalog=session.catalog
        ).save()

        assert isinstance(ds_tmp, DatasetQuery)
        assert ds_tmp.name != ds_name
        assert ds_tmp.name.startswith(Session.DATASET_PREFIX)
        assert session_name in ds_tmp.name

        ds = catalog.get_dataset(ds_tmp.name)
        assert ds is not None

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(ds_tmp.name)
