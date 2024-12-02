import sys
from textwrap import dedent

import cloudpickle
import pytest

from datachain.cli import query
from datachain.data_storage import AbstractDBMetastore, JobQueryType, JobStatus
from datachain.job import Job


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
    from datachain import DataChain, metrics, param

    dc = DataChain.from_storage(param("url"), session=session)

    metrics.set("count", dc.count())

    dc.save("my-ds")
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
    job = get_latest_job(catalog.metastore)
    assert job.id == result_job_id
    assert job.name == "query_script.py"
    assert job.status == JobStatus.COMPLETE
    assert job.query == query_script
    assert job.query_type == JobQueryType.PYTHON
    assert job.workers == 1
    assert job.params == {"url": src_uri}
    assert job.metrics == {"count": 7}
    assert job.python_version == f"{sys.version_info.major}.{sys.version_info.minor}"
