import os.path
import signal
import sys
from multiprocessing.pool import ExceptionWithTraceback  # type: ignore[attr-defined]
from textwrap import dedent

import cloudpickle
import multiprocess
import pytest

from datachain.catalog import Catalog
from datachain.cli import query
from datachain.data_storage import AbstractDBMetastore, JobQueryType, JobStatus
from datachain.error import QueryScriptCancelError
from datachain.job import Job
from tests.utils import wait_for_condition


@pytest.fixture
def catalog_info_filepath(cloud_test_catalog_tmpfile, tmp_path):
    catalog = cloud_test_catalog_tmpfile.catalog

    catalog_info = {
        "catalog_init_params": catalog.get_init_params(),
        "metastore_params": catalog.metastore.clone_params(),
        "warehouse_params": catalog.warehouse.clone_params(),
    }
    catalog_info_filepath = tmp_path / "catalog-info"
    with open(catalog_info_filepath, "wb") as f:
        cloudpickle.dump(catalog_info, f)

    return catalog_info_filepath


def setup_catalog(query: str, catalog_info_filepath: str) -> str:
    query_catalog_setup = f"""\
    import cloudpickle
    from datachain.catalog import Catalog
    from datachain.query.session import Session

    catalog_info_filepath = {str(catalog_info_filepath)!r}
    with open(catalog_info_filepath, "rb") as f:
        catalog_info = cloudpickle.load(f)
    (
        metastore_class,
        metastore_args,
        metastore_kwargs,
    ) = catalog_info["metastore_params"]
    metastore = metastore_class(*metastore_args, **metastore_kwargs)
    (
        warehouse_class,
        warehouse_args,
        warehouse_kwargs,
    ) = catalog_info["warehouse_params"]
    warehouse = warehouse_class(*warehouse_args, **warehouse_kwargs)
    catalog = Catalog(
        metastore=metastore,
        warehouse=warehouse,
        **catalog_info["catalog_init_params"],
    )
    session = Session("test", catalog=catalog)
    """
    return dedent(query_catalog_setup + "\n" + query)


def get_latest_job(metastore: AbstractDBMetastore) -> Job:
    j = metastore._jobs
    query = metastore._jobs_select().order_by(j.c.created_at.desc()).limit(1)
    (row,) = metastore.db.execute(query)
    return metastore._parse_job(row)


@pytest.mark.parametrize("cloud_type,version_aware", [("file", False)], indirect=True)
@pytest.mark.xdist_group(name="tmpfile")
def test_query_cli(cloud_test_catalog_tmpfile, tmp_path, catalog_info_filepath, capsys):
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri

    query_script = """\
    import datachain as dc
    from datachain import metrics, param

    chain = dc.read_storage(param("url"), session=session)

    metrics.set("count", chain.count())

    chain.save("my-ds")
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)

    filepath = tmp_path / "query_script.py"
    filepath.write_text(query_script)

    query(catalog, str(filepath), params={"url": src_uri})

    out, err = capsys.readouterr()
    assert not out
    assert not err

    dataset = catalog.get_dataset("my-ds")
    result_job_id = dataset.get_version(dataset.latest_version).job_id
    assert result_job_id
    if not os.environ.get("DATACHAIN_DISTRIBUTED"):
        job = get_latest_job(catalog.metastore)
        assert job.id == result_job_id
        assert job.name == "query_script.py"
        assert job.status == JobStatus.COMPLETE
        assert job.query == query_script
        assert job.query_type == JobQueryType.PYTHON
        assert job.workers == 1
        assert job.params == {"url": src_uri}
        assert job.metrics == {"count": 7}
        assert (
            job.python_version == f"{sys.version_info.major}.{sys.version_info.minor}"
        )


if sys.platform == "win32":
    SIGKILL = signal.SIGTERM
else:
    SIGKILL = signal.SIGKILL


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not have SIGTERM")
@pytest.mark.parametrize(
    "setup,expected_return_code",
    [
        ("", -signal.SIGINT),
        ("signal.signal(signal.SIGINT, signal.SIG_IGN)", -signal.SIGTERM),
        (
            """\
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
""",
            -SIGKILL,
        ),
    ],
)
def test_shutdown_on_sigterm(tmp_dir, request, catalog, setup, expected_return_code):
    query = f"""\
import os, pathlib, signal, sys, time

pathlib.Path("ready").touch(exist_ok=False)
{setup}
time.sleep(10)
"""

    def apply(f, args, kwargs):
        return f(*args, **kwargs)

    def func(ms_params, wh_params, init_params, q):
        catalog = Catalog(apply(*ms_params), apply(*wh_params), **init_params)
        try:
            catalog.query(query, interrupt_timeout=0.5, terminate_timeout=0.5)
        except Exception as e:  # noqa: BLE001
            q.put(ExceptionWithTraceback(e, e.__traceback__))
        else:
            q.put(None)

    mp_ctx = multiprocess.get_context("spawn")
    q = mp_ctx.Queue()
    p = mp_ctx.Process(
        target=func,
        args=(
            catalog.metastore.clone_params(),
            catalog.warehouse.clone_params(),
            catalog.get_init_params(),
            q,
        ),
    )
    p.start()
    request.addfinalizer(p.kill)

    def is_ready():
        assert p.is_alive(), "Process is dead"
        return os.path.exists("ready")

    # make sure the process is running before we send the signal
    wait_for_condition(is_ready, "script to start", timeout=5)

    os.kill(p.pid, signal.SIGTERM)
    p.join(timeout=3)  # might take as long as 1 second to complete shutdown_process
    assert not p.exitcode

    e = q.get_nowait()
    assert isinstance(e, QueryScriptCancelError)
    assert e.return_code == expected_return_code
