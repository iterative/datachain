import os
import os.path
import uuid
from collections.abc import Generator
from pathlib import PosixPath
from typing import NamedTuple

import attrs
import pytest
import sqlalchemy
from pytest import MonkeyPatch, TempPathFactory
from upath.implementations.cloud import CloudPath

from datachain.catalog import Catalog
from datachain.catalog.loader import get_id_generator, get_metastore, get_warehouse
from datachain.cli_utils import CommaSeparatedArgs
from datachain.config import Config, ConfigLevel
from datachain.data_storage.sqlite import (
    SQLiteDatabaseEngine,
    SQLiteIDGenerator,
    SQLiteMetastore,
    SQLiteWarehouse,
)
from datachain.dataset import DatasetRecord
from datachain.lib.dc import DataChain, Sys
from datachain.query.session import Session
from datachain.utils import (
    ENV_DATACHAIN_GLOBAL_CONFIG_DIR,
    ENV_DATACHAIN_SYSTEM_CONFIG_DIR,
    STUDIO_URL,
    DataChainDir,
)

from .utils import DEFAULT_TREE, instantiate_tree

DEFAULT_DATACHAIN_BIN = "datachain"
DEFAULT_DATACHAIN_GIT_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

collect_ignore = ["setup.py"]


@pytest.fixture(scope="session", autouse=True)
def add_test_env():
    os.environ["DATACHAIN_TEST"] = "true"


@pytest.fixture(autouse=True)
def global_config_dir(monkeypatch, tmp_path_factory):
    global_dir = str(tmp_path_factory.mktemp("studio-login-global"))
    monkeypatch.setenv(ENV_DATACHAIN_GLOBAL_CONFIG_DIR, global_dir)
    yield global_dir


@pytest.fixture(autouse=True)
def system_config_dir(monkeypatch, tmp_path_factory):
    system_dir = str(tmp_path_factory.mktemp("studio-login-system"))
    monkeypatch.setenv(ENV_DATACHAIN_SYSTEM_CONFIG_DIR, system_dir)
    yield system_dir


@pytest.fixture(autouse=True)
def virtual_memory(mocker):
    class VirtualMemory(NamedTuple):
        total: int
        available: int
        percent: int
        used: int
        free: int

    return mocker.patch(
        "psutil.virtual_memory",
        return_value=VirtualMemory(
            total=1073741824,
            available=5368709120,
            # prevent dumping of smaller batches to db in process_udf_outputs
            # we want to avoid this due to tests running concurrently (xdist)
            percent=50,
            used=5368709120,
            free=5368709120,
        ),
    )


@pytest.fixture(autouse=True)
def per_thread_im_mem_db(mocker, worker_id):
    if worker_id == "master":
        return
    mocker.patch(
        "datachain.data_storage.sqlite._get_in_memory_uri",
        return_value=f"file:in-mem-db-{worker_id}?mode=memory&cache=shared",
    )


@pytest.fixture(scope="session")
def monkeypatch_session() -> Generator[MonkeyPatch, None, None]:
    """
    Like monkeypatch, but for session scope.
    """
    mpatch = pytest.MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(autouse=True)
def clean_session() -> None:
    """
    Make sure we clean leftover session before each test case
    """
    Session.cleanup_for_tests()


@pytest.fixture(scope="session", autouse=True)
def clean_environment(
    monkeypatch_session: MonkeyPatch,
    tmp_path_factory: TempPathFactory,
) -> None:
    """
    Make sure we have a clean environment and won't write to userspace.
    """
    working_dir = str(tmp_path_factory.mktemp("default_working_dir"))
    monkeypatch_session.chdir(working_dir)
    monkeypatch_session.delenv(DataChainDir.ENV_VAR, raising=False)


@pytest.fixture
def sqlite_db():
    if os.environ.get("DATACHAIN_METASTORE") or os.environ.get("DATACHAIN_WAREHOUSE"):
        pytest.skip("This test only runs on SQLite")
    with SQLiteDatabaseEngine.from_db_file(":memory:") as db:
        yield db
        cleanup_sqlite_db(db, [])


def cleanup_sqlite_db(
    db: SQLiteDatabaseEngine,
    cleanup_tables: list[str],
):
    # Wipe the DB after the test
    # Using new connection to check that the DB isn't locked
    tables = db.execute_str(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    # removing in reversed order because of foreign keys
    for table in reversed(cleanup_tables):
        db.execute_str(f"DROP TABLE IF EXISTS '{table}'")

    for (table,) in tables:
        name = table.replace("'", "''")
        db.execute_str(f"DROP TABLE IF EXISTS '{name}'")

    # Close the connection so that the SQLite file is no longer open, to avoid
    # pytest throwing: OSError: [Errno 24] Too many open files
    db.close()


@pytest.fixture
def id_generator():
    if os.environ.get("DATACHAIN_ID_GENERATOR"):
        _id_generator = get_id_generator()
        yield _id_generator

        _id_generator.cleanup_for_tests()
    else:
        db = SQLiteDatabaseEngine.from_db_file(":memory:")
        _id_generator = SQLiteIDGenerator(db)
        yield _id_generator

        _id_generator.cleanup_for_tests()

    # Close the connection so that the SQLite file is no longer open, to avoid
    # pytest throwing: OSError: [Errno 24] Too many open files
    # Or, for other implementations, prevent "too many clients" errors.
    _id_generator.close_on_exit()


@pytest.fixture
def metastore(id_generator):
    if os.environ.get("DATACHAIN_METASTORE"):
        _metastore = get_metastore(id_generator)
        yield _metastore

        _metastore.cleanup_for_tests()
    else:
        _metastore = SQLiteMetastore(id_generator, db_file=":memory:")
        yield _metastore

        cleanup_sqlite_db(_metastore.db.clone(), _metastore.default_table_names)

    # Close the connection so that the SQLite file is no longer open, to avoid
    # pytest throwing: OSError: [Errno 24] Too many open files
    # Or, for other implementations, prevent "too many clients" errors.
    _metastore.close_on_exit()


def check_temp_tables_cleaned_up(original_warehouse):
    """Ensure that temporary tables are cleaned up."""
    with original_warehouse.clone() as warehouse:
        assert [
            t
            for t in sqlalchemy.inspect(warehouse.db.engine).get_table_names()
            if t.startswith(
                (warehouse.UDF_TABLE_NAME_PREFIX, warehouse.TMP_TABLE_NAME_PREFIX)
            )
        ] == []


@pytest.fixture
def warehouse(id_generator, metastore):
    if os.environ.get("DATACHAIN_WAREHOUSE"):
        _warehouse = get_warehouse(id_generator)
        yield _warehouse
        try:
            check_temp_tables_cleaned_up(_warehouse)
        finally:
            _warehouse.cleanup_for_tests()
    else:
        _warehouse = SQLiteWarehouse(id_generator, db_file=":memory:")
        yield _warehouse
        try:
            check_temp_tables_cleaned_up(_warehouse)
        finally:
            cleanup_sqlite_db(_warehouse.db.clone(), metastore.default_table_names)

            # Close the connection so that the SQLite file is no longer open, to avoid
            # pytest throwing: OSError: [Errno 24] Too many open files
            _warehouse.db.close()


@pytest.fixture
def catalog(id_generator, metastore, warehouse):
    return Catalog(id_generator=id_generator, metastore=metastore, warehouse=warehouse)


@pytest.fixture
def test_session(catalog):
    with Session("TestSession", catalog=catalog) as session:
        yield session


@pytest.fixture
def id_generator_tmpfile(tmp_path):
    if os.environ.get("DATACHAIN_ID_GENERATOR"):
        _id_generator = get_id_generator()
        yield _id_generator

        _id_generator.cleanup_for_tests()
    else:
        db = SQLiteDatabaseEngine.from_db_file(tmp_path / "test.db")
        _id_generator = SQLiteIDGenerator(db)
        yield _id_generator

        _id_generator.cleanup_for_tests()

    # Close the connection so that the SQLite file is no longer open, to avoid
    # pytest throwing: OSError: [Errno 24] Too many open files
    # Or, for other implementations, prevent "too many clients" errors.
    _id_generator.close_on_exit()


@pytest.fixture
def metastore_tmpfile(tmp_path, id_generator_tmpfile):
    if os.environ.get("DATACHAIN_METASTORE"):
        _metastore = get_metastore(id_generator_tmpfile)
        yield _metastore

        _metastore.cleanup_for_tests()
    else:
        _metastore = SQLiteMetastore(id_generator_tmpfile, db_file=tmp_path / "test.db")
        yield _metastore

        cleanup_sqlite_db(_metastore.db.clone(), _metastore.default_table_names)

    # Close the connection so that the SQLite file is no longer open, to avoid
    # pytest throwing: OSError: [Errno 24] Too many open files
    # Or, for other implementations, prevent "too many clients" errors.
    _metastore.close_on_exit()


@pytest.fixture
def warehouse_tmpfile(tmp_path, id_generator_tmpfile, metastore_tmpfile):
    if os.environ.get("DATACHAIN_WAREHOUSE"):
        _warehouse = get_warehouse(id_generator_tmpfile)
        yield _warehouse
        try:
            check_temp_tables_cleaned_up(_warehouse)
        finally:
            _warehouse.cleanup_for_tests()
    else:
        _warehouse = SQLiteWarehouse(id_generator_tmpfile, db_file=tmp_path / "test.db")
        yield _warehouse
        try:
            check_temp_tables_cleaned_up(_warehouse)
        finally:
            cleanup_sqlite_db(
                _warehouse.db.clone(), metastore_tmpfile.default_table_names
            )

            # Close the connection so that the SQLite file is no longer open, to avoid
            # pytest throwing: OSError: [Errno 24] Too many open files
            _warehouse.db.close()


@pytest.fixture
def catalog_tmpfile(id_generator_tmpfile, metastore_tmpfile, warehouse_tmpfile):
    # For testing parallel and distributed processing, as these cannot use
    # in-memory databases.
    return Catalog(
        id_generator=id_generator_tmpfile,
        metastore=metastore_tmpfile,
        warehouse=warehouse_tmpfile,
    )


@pytest.fixture
def test_session_tmpfile(catalog_tmpfile):
    # For testing parallel and distributed processing, as these cannot use
    # in-memory databases.
    return Session("TestSession", catalog=catalog_tmpfile)


@pytest.fixture
def tmp_dir(tmp_path_factory, monkeypatch):
    dpath = tmp_path_factory.mktemp("datachain-test")
    monkeypatch.chdir(dpath)
    return dpath


def pytest_addoption(parser):
    parser.addoption(
        "--disable-remotes",
        action=CommaSeparatedArgs,
        default=[],
        help="Comma separated list of remotes to disable",
    )

    parser.addoption(
        "--datachain-bin",
        type=str,
        default=DEFAULT_DATACHAIN_BIN,
        help="Path to datachain binary",
    )

    parser.addoption(
        "--datachain-revs",
        type=str,
        help="Comma-separated list of DataChain revisions to test "
        "(overrides `--datachain-bin`)",
    )

    parser.addoption(
        "--datachain-git-repo",
        type=str,
        default=DEFAULT_DATACHAIN_GIT_REPO,
        help="Path or url to datachain git repo",
    )
    parser.addoption(
        "--azure-connection-string",
        type=str,
        help=(
            "Connection string to run tests against a real, versioned "
            "Azure storage account"
        ),
    )


class DataChainTestConfig:
    def __init__(self):
        self.datachain_bin = DEFAULT_DATACHAIN_BIN
        self.datachain_revs = None
        self.datachain_git_repo = DEFAULT_DATACHAIN_GIT_REPO


@pytest.fixture(scope="session")
def test_config(request):
    return request.config.datachain_config


def pytest_configure(config):
    config.datachain_config = DataChainTestConfig()

    config.datachain_config.datachain_bin = config.getoption("--datachain-bin")
    config.datachain_config.datachain_revs = config.getoption("--datachain-revs")
    config.datachain_config.datachain_git_repo = config.getoption(
        "--datachain-git-repo"
    )


@pytest.fixture(scope="session", params=[DEFAULT_TREE])
def tree(request):
    return request.param


@attrs.define
class CloudServer:
    kind: str
    src: CloudPath
    client_config: dict[str, str]

    @property
    def src_uri(self):
        if self.kind == "file":
            return self.src.as_uri()
        return str(self.src).rstrip("/")


def make_cloud_server(src_path, cloud_type, tree):
    fs = src_path.fs
    if cloud_type == "s3":
        endpoint_url = fs.client_kwargs["endpoint_url"]
        client_config = {"aws_endpoint_url": endpoint_url}
    elif cloud_type in ("gs", "gcs"):
        endpoint_url = fs._endpoint
        client_config = {"endpoint_url": endpoint_url}
    elif cloud_type == "azure":
        client_config = fs.storage_options.copy()
    elif cloud_type == "file":
        client_config = {}
    else:
        raise ValueError(f"invalid cloud_type: {cloud_type}")

    instantiate_tree(src_path, tree)
    return CloudServer(kind=cloud_type, src=src_path, client_config=client_config)


@attrs.define
class CloudTestCatalog:
    server: CloudServer
    working_dir: PosixPath
    catalog: Catalog

    @property
    def src(self):
        return self.server.src

    @property
    def src_uri(self):
        return self.server.src_uri

    @property
    def client_config(self):
        return self.server.client_config

    @property
    def session(self) -> Session:
        return Session("CTCSession", catalog=self.catalog)


cloud_types = ["s3", "gs", "azure"]


@pytest.fixture(scope="session", params=["file", *cloud_types])
def cloud_type(request):
    return request.param


@pytest.fixture(scope="session", params=[False, True])
def version_aware(request):
    return request.param


def pytest_collection_modifyitems(config, items):
    disabled_remotes = config.getoption("--disable-remotes")
    if not disabled_remotes:
        return

    for item in items:
        if "cloud_server" in item.fixturenames:
            cloud_type = item.callspec.params.get("cloud_type")
            if cloud_type not in cloud_types:
                continue
            if "all" in disabled_remotes:
                reason = "Skipping all tests requiring cloud"
                item.add_marker(pytest.mark.skip(reason=reason))
            if cloud_type in disabled_remotes:
                reason = f"Skipping all tests for {cloud_type=}"
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def cloud_server(request, tmp_upath_factory, cloud_type, version_aware, tree):
    if cloud_type == "azure" and version_aware:
        if conn_str := request.config.getoption("--azure-connection-string"):
            src_path = tmp_upath_factory.azure(conn_str)
        else:
            pytest.skip("Can't test versioning with Azure")
    elif cloud_type == "file":
        if version_aware:
            pytest.skip("Local storage can't be versioned")
        else:
            src_path = tmp_upath_factory.mktemp("local")
    else:
        src_path = tmp_upath_factory.mktemp(cloud_type, version_aware=version_aware)
    return make_cloud_server(src_path, cloud_type, tree)


@pytest.fixture()
def datachain_job_id(monkeypatch):
    job_id = uuid.uuid4().hex
    monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)
    return job_id


@pytest.fixture
def cloud_server_credentials(cloud_server, monkeypatch):
    if cloud_server.kind == "s3":
        cfg = cloud_server.src.fs.client_kwargs
        try:
            monkeypatch.delenv("AWS_PROFILE")
        except KeyError:
            pass
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", cfg.get("aws_access_key_id"))
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", cfg.get("aws_secret_access_key"))
        monkeypatch.setenv("AWS_SESSION_TOKEN", cfg.get("aws_session_token"))
        monkeypatch.setenv("AWS_DEFAULT_REGION", cfg.get("region_name"))


def get_cloud_test_catalog(cloud_server, tmp_path, id_generator, metastore, warehouse):
    cache_dir = tmp_path / ".datachain" / "cache"
    cache_dir.mkdir(parents=True)
    tmpfile_dir = tmp_path / ".datachain" / "tmp"
    tmpfile_dir.mkdir()

    catalog = Catalog(
        id_generator=id_generator,
        metastore=metastore,
        warehouse=warehouse,
        cache_dir=str(cache_dir),
        tmp_dir=str(tmpfile_dir),
        client_config=cloud_server.client_config,
    )
    return CloudTestCatalog(server=cloud_server, working_dir=tmp_path, catalog=catalog)


@pytest.fixture
def cloud_test_catalog(
    cloud_server,
    cloud_server_credentials,
    tmp_path,
    id_generator,
    metastore,
    warehouse,
):
    return get_cloud_test_catalog(
        cloud_server, tmp_path, id_generator, metastore, warehouse
    )


@pytest.fixture
def cloud_test_catalog_upload(cloud_test_catalog):
    """This returns a version of the cloud_test_catalog that is suitable for uploading
    files, and will perform the necessary cleanup of any uploaded files."""
    from datachain.client.fsspec import Client

    src = cloud_test_catalog.src_uri
    client = Client.get_implementation(src)
    fsspec_fs = client.create_fs(**cloud_test_catalog.client_config)
    original_paths = set(fsspec_fs.ls(src))

    yield cloud_test_catalog

    # Cleanup any written files
    new_paths = set(fsspec_fs.ls(src))
    cleanup_paths = new_paths - original_paths
    for p in cleanup_paths:
        fsspec_fs.rm(p, recursive=True)


@pytest.fixture
def cloud_test_catalog_tmpfile(
    cloud_server,
    cloud_server_credentials,
    tmp_path,
    id_generator_tmpfile,
    metastore_tmpfile,
    warehouse_tmpfile,
):
    return get_cloud_test_catalog(
        cloud_server,
        tmp_path,
        id_generator_tmpfile,
        metastore_tmpfile,
        warehouse_tmpfile,
    )


@pytest.fixture
def listed_bucket(cloud_test_catalog):
    ctc = cloud_test_catalog
    DataChain.from_storage(ctc.src_uri, session=ctc.session).exec()


@pytest.fixture
def animal_dataset(listed_bucket, cloud_test_catalog):
    name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    dataset = catalog.create_dataset_from_sources(name, [src_uri], recursive=True)
    return catalog.update_dataset(
        dataset, {"description": "animal dataset", "labels": ["cats", "dogs"]}
    )


@pytest.fixture
def dogs_dataset(listed_bucket, cloud_test_catalog):
    name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    dataset = catalog.create_dataset_from_sources(
        name, [f"{src_uri}/dogs/*"], recursive=True
    )
    return catalog.update_dataset(
        dataset, {"description": "dogs dataset", "labels": ["dogs", "dataset"]}
    )


@pytest.fixture
def cats_dataset(listed_bucket, cloud_test_catalog):
    name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    dataset = catalog.create_dataset_from_sources(
        name, [f"{src_uri}/cats/*"], recursive=True
    )
    return catalog.update_dataset(
        dataset, {"description": "cats dataset", "labels": ["cats", "dataset"]}
    )


@pytest.fixture
def dataset_record():
    return DatasetRecord(
        id=1,
        name=f"ds_{uuid.uuid4().hex}",
        description="",
        labels=[],
        versions=[],
        status=1,
        schema={},
        feature_schema={},
    )


@pytest.fixture
def dataset_rows():
    int_example = 25
    return [
        {
            "id": i,
            "location": "",
            "source": "s3://my-bucket",
            "dir_type": 0,
            "parent": "input/text_emd_1m",
            "version": "7e589b7d-382c-49a5-931f-2b999c930c5e",
            "is_latest": True,
            "name": f"dql_1m_meta_text_emd.parquet_3_{i}_0.snappy.parquet",
            "etag": f"72b35c8e9b8eed1636c91eb94241c2f8-{i}",
            "last_modified": "2024-02-23T10:42:31.842944+00:00",
            "size": 49807360,
            "sys__rand": 12123123123,
            "int_col": 5,
            "int_col_32": 5,
            "int_col_64": 5,
            "float_col": 0.5,
            "float_col_32": 0.5,
            "float_col_64": 0.5,
            "array_col": [0.5],
            "array_col_nested": [[0.5]],
            "array_col_32": [0.5],
            "array_col_64": [0.5],
            "string_col": "a string",
            "bool_col": True,
            "json_col": '{"a": 1}',
            "binary_col": int_example.to_bytes(2, "big"),
        }
        for i in range(19)
    ]


@pytest.fixture
def studio_datasets(requests_mock):
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"token": "isat_access_token", "team": "team_name"}

    datasets = [
        {
            "id": 1,
            "name": "dogs",
            "versions": [{"version": 1}, {"version": 2}],
        },
        {
            "id": 2,
            "name": "cats",
            "versions": [{"version": 1}],
        },
        {
            "id": 3,
            "name": "both",
            "versions": [{"version": 1}],
        },
    ]

    requests_mock.post(f"{STUDIO_URL}/api/datachain/ls-datasets", json=datasets)


@pytest.fixture
def not_random_ds(test_session):
    return DataChain.from_records(
        [
            {"sys__id": 1, "sys__rand": 200000000, "fib": 0},
            {"sys__id": 2, "sys__rand": 400000000, "fib": 1},
            {"sys__id": 3, "sys__rand": 600000000, "fib": 1},
            {"sys__id": 4, "sys__rand": 800000000, "fib": 2},
            {"sys__id": 5, "sys__rand": 1000000000, "fib": 3},
            {"sys__id": 6, "sys__rand": 1200000000, "fib": 5},
            {"sys__id": 7, "sys__rand": 1400000000, "fib": 8},
            {"sys__id": 8, "sys__rand": 1600000000, "fib": 13},
            {"sys__id": 9, "sys__rand": 1800000000, "fib": 21},
            {"sys__id": 10, "sys__rand": 2000000000, "fib": 34},
        ],
        session=test_session,
        schema={"sys": Sys, "fib": int},
    )


@pytest.fixture
def pseudo_random_ds(test_session):
    return DataChain.from_records(
        [
            {"sys__id": 1, "sys__rand": 2406827533654413759, "fib": 0},
            {"sys__id": 2, "sys__rand": 743035223448130834, "fib": 1},
            {"sys__id": 3, "sys__rand": 8572034894545971037, "fib": 1},
            {"sys__id": 4, "sys__rand": 3413911135601125438, "fib": 2},
            {"sys__id": 5, "sys__rand": 8036488725627198326, "fib": 3},
            {"sys__id": 6, "sys__rand": 2020789040280779494, "fib": 5},
            {"sys__id": 7, "sys__rand": 8478782014085172114, "fib": 8},
            {"sys__id": 8, "sys__rand": 1374262678671783922, "fib": 13},
            {"sys__id": 9, "sys__rand": 7728884931956308771, "fib": 21},
            {"sys__id": 10, "sys__rand": 5591681088079559562, "fib": 34},
        ],
        session=test_session,
        schema={"sys": Sys, "fib": int},
    )
