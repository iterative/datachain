from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func

from datachain import utils
from datachain.error import StorageNotFoundError
from datachain.storage import STALE_MINUTES_LIMIT, Storage, StorageStatus, StorageURI
from tests.utils import skip_if_not_sqlite

TS = datetime(2022, 8, 1)
EXPIRES = datetime(2022, 8, 2)


def test_human_time():
    assert utils.human_time_to_int("1236") == 1236
    assert utils.human_time_to_int("3h") == 3 * 60 * 60
    assert utils.human_time_to_int("2w") == 2 * 7 * 24 * 60 * 60
    assert utils.human_time_to_int("4M") == 4 * 31 * 24 * 60 * 60

    assert utils.human_time_to_int("bla") is None


def test_storage():
    s = Storage("s3://foo", TS, EXPIRES)

    d = s.to_dict()
    assert d.get("uri") == s.uri


def test_expiration_time():
    assert Storage.get_expiration_time(TS, 12344) == TS + timedelta(seconds=12344)


def test_adding_storage(metastore):
    uri = StorageURI("s3://whatever")
    with pytest.raises(StorageNotFoundError):
        metastore.get_storage(uri)

    storage, _, _, _, _ = metastore.register_storage_for_indexing(uri)
    cnt = next(metastore.db.execute(metastore._storages_select(func.count())), (0,))
    assert cnt[0] == 1

    bkt = list(
        metastore.db.execute(
            metastore._storages_select().where(metastore._storages.c.uri == uri)
        )
    )
    assert len(bkt) == 1

    assert storage.id == bkt[0][0]
    assert storage.uri == bkt[0][1]
    assert storage.timestamp == bkt[0][2]
    assert storage.expires == bkt[0][3]
    assert storage.started_inserting_at == bkt[0][4]
    assert storage.last_inserted_at == bkt[0][5]
    assert storage.status == bkt[0][6]
    assert storage.error_message == ""
    assert storage.error_stack == ""


def test_storage_status(metastore):
    uri = StorageURI("s3://somebucket")

    metastore.create_storage_if_not_registered(uri)
    storage = metastore.get_storage(uri)
    assert storage.uri == uri
    assert storage.status == StorageStatus.CREATED

    (
        storage,
        need_index,
        in_progress,
        partial_id,
        partial_path,
    ) = metastore.register_storage_for_indexing(uri)
    assert storage.status == StorageStatus.PENDING
    assert storage.uri == uri
    assert storage == metastore.get_storage(uri)
    assert need_index is True
    assert in_progress is False
    assert partial_id is None
    assert partial_path is None

    (
        s2,
        need_index,
        in_progress,
        partial_id,
        partial_path,
    ) = metastore.register_storage_for_indexing(uri)
    assert s2.status == StorageStatus.PENDING
    assert storage == s2 == metastore.get_storage(uri)
    assert need_index is False
    assert in_progress is True
    assert partial_id is None
    assert partial_path is None

    end_time = datetime.now(timezone.utc)
    metastore.mark_storage_indexed(uri, StorageStatus.COMPLETE, 1000, end_time)
    storage = metastore.get_storage(uri)
    assert storage.status == StorageStatus.COMPLETE


@pytest.mark.parametrize(
    "ttl",
    (-1, 999999999999, 99999999999999, 9999999999999999),
)
def test_max_ttl(ttl):
    uri = "s3://whatever"
    expires = Storage.get_expiration_time(TS, ttl)
    storage = Storage(1, uri, TS, expires)
    assert storage.timestamp == TS
    assert storage.expires == datetime.max
    assert storage.timestamp_str  # no error
    assert storage.timestamp_to_local  # no error
    assert storage.expires_to_local  # no error


def test_storage_without_dates():
    uri = "s3://whatever"
    storage = Storage(1, uri, None, None)
    assert storage.timestamp is None
    assert storage.expires is None
    assert storage.timestamp_str is None  # no error
    assert storage.timestamp_to_local is None  # no error
    assert storage.expires_to_local is None  # no error
    assert storage.to_dict() == {
        "uri": uri,
        "timestamp": None,
        "expires": None,
    }


def test_storage_update_last_inserted_at(metastore):
    uri = StorageURI("s3://bucket_last_inserted")
    metastore.create_storage_if_not_registered(uri)
    metastore.update_last_inserted_at(uri)
    storage = metastore.get_storage(uri)
    assert storage.last_inserted_at


def test_stale_storage(metastore):
    uri_stale = StorageURI("s3://bucket_stale")
    uri_not_stale = StorageURI("s3://bucket_not_stale")

    metastore.create_storage_if_not_registered(uri_stale)
    metastore.create_storage_if_not_registered(uri_not_stale)

    metastore.mark_storage_pending(metastore.get_storage(uri_stale))
    metastore.mark_storage_pending(metastore.get_storage(uri_not_stale))

    # make storage looks stale
    updates = {
        "last_inserted_at": datetime.now(timezone.utc)
        - timedelta(minutes=STALE_MINUTES_LIMIT + 1)
    }
    s = metastore._storages
    metastore.db.execute(s.update().where(s.c.uri == uri_stale).values(**updates))

    metastore.find_stale_storages()

    stale_storage = metastore.get_storage(uri_stale)
    assert stale_storage.status == StorageStatus.STALE

    not_stale_storage = metastore.get_storage(uri_not_stale)
    assert not_stale_storage.status == StorageStatus.PENDING


def test_failed_storage(metastore):
    uri = StorageURI("s3://bucket")
    error_message = "Internal error on indexing"
    error_stack = "error"
    metastore.create_storage_if_not_registered(uri)

    metastore.mark_storage_pending(metastore.get_storage(uri))
    metastore.mark_storage_indexed(
        uri,
        StorageStatus.FAILED,
        1000,
        datetime.now(),
        error_message=error_message,
        error_stack=error_stack,
    )

    storage = metastore.get_storage(uri)
    assert storage.status == StorageStatus.FAILED
    assert storage.error_message == error_message
    assert storage.error_stack == error_stack


def test_unlist_source(
    listed_bucket,
    cloud_test_catalog,
    cloud_type,
):
    # TODO remove when https://github.com/iterative/dvcx/pull/868 is merged
    skip_if_not_sqlite()
    source_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog
    _partial_id, partial_path = catalog.metastore.get_valid_partial_id(
        cloud_test_catalog.storage_uri, cloud_test_catalog.partial_path
    )
    storage_dataset_name = Storage.dataset_name(
        cloud_test_catalog.storage_uri, partial_path
    )

    # list source
    storage = catalog.get_storage(cloud_test_catalog.storage_uri)
    if cloud_type == "file":
        assert storage.status == StorageStatus.PARTIAL
    else:
        assert storage.status == StorageStatus.COMPLETE

    catalog.get_dataset(storage_dataset_name)

    # unlist source
    catalog.unlist_source(source_uri)
    with pytest.raises(StorageNotFoundError):
        catalog.get_storage(source_uri)
    # we preserve the table for dataset lineage
    catalog.get_dataset(storage_dataset_name)
