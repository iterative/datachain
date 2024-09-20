import os.path
from textwrap import dedent
from typing import TYPE_CHECKING

import dill
import pytest

from datachain.cli import query
from datachain.data_storage import AbstractDBMetastore, JobQueryType, JobStatus
from tests.utils import assert_row_names

if TYPE_CHECKING:
    from datachain.job import Job


@pytest.fixture
def catalog_info_filepath(cloud_test_catalog_tmpfile, tmp_path):
    catalog = cloud_test_catalog_tmpfile.catalog

    catalog_info = {
        "catalog_init_params": catalog.get_init_params(),
        "id_generator_params": catalog.id_generator.clone_params(),
        "metastore_params": catalog.metastore.clone_params(),
        "warehouse_params": catalog.warehouse.clone_params(),
    }
    catalog_info_filepath = tmp_path / "catalog-info"
    with open(catalog_info_filepath, "wb") as f:
        dill.dump(catalog_info, f)

    return catalog_info_filepath


def setup_catalog(query: str, catalog_info_filepath: str) -> str:
    query_catalog_setup = f"""\
    import dill
    from datachain.catalog import Catalog

    catalog_info_filepath = {str(catalog_info_filepath)!r}
    with open(catalog_info_filepath, "rb") as f:
        catalog_info = dill.load(f)
    (
        id_generator_class,
        id_generator_args,
        id_generator_kwargs,
    ) = catalog_info["id_generator_params"]
    id_generator = id_generator_class(*id_generator_args, **id_generator_kwargs)
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
        id_generator=id_generator,
        metastore=metastore,
        warehouse=warehouse,
        **catalog_info["catalog_init_params"],
    )
    """
    return dedent(query_catalog_setup + "\n" + query)


def get_latest_job(
    metastore: AbstractDBMetastore,
) -> "Job":
    j = metastore._jobs
    query = metastore._jobs_select().order_by(j.c.created_at.desc()).limit(1)
    (row,) = metastore.db.execute(query)
    return metastore._parse_job(row)


@pytest.mark.parametrize("cloud_type,version_aware", [("file", False)], indirect=True)
@pytest.mark.xdist_group(name="tmpfile")
def test_query_cli(cloud_test_catalog_tmpfile, tmp_path, catalog_info_filepath, capsys):
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri

    query_script = f"""\
    from datachain.query import DatasetQuery
    from datachain import C
    from datachain.sql.functions.path import name

    catalog.create_dataset_from_sources("animals", ["{src_uri}"], recursive=True)

    DatasetQuery("animals", catalog=catalog).mutate(
        name=name(C("file__path"))
    ).save("my-ds")
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)

    filepath = tmp_path / "query_script.py"
    filepath.write_text(query_script)

    query(catalog, str(filepath))

    out, err = capsys.readouterr()
    assert not out
    assert not err

    dataset = catalog.get_dataset("my-ds")
    assert dataset
    result_job_id = dataset.get_version(dataset.latest_version).job_id
    assert result_job_id

    latest_job = get_latest_job(catalog.metastore)
    assert latest_job

    assert str(latest_job.id) == str(result_job_id)
    assert latest_job.name == os.path.basename(filepath)
    assert latest_job.status == JobStatus.COMPLETE
    assert latest_job.query_type == JobQueryType.PYTHON
    assert latest_job.error_message == ""
    assert latest_job.error_stack == ""


@pytest.mark.xdist_group(name="tmpfile")
def test_query(cloud_test_catalog_tmpfile, tmp_path, catalog_info_filepath):
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri
    catalog.create_dataset_from_sources("animals", [src_uri], recursive=True)

    query_script = """\
    from datachain.query import DatasetQuery, C
    DatasetQuery("animals", catalog=catalog).filter(
        C("file__path").glob("*dog*")
    ).save("my-ds")
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)
    catalog.query(query_script)

    dataset = catalog.get_dataset("my-ds")
    assert dataset.versions_values == [1]
    assert_row_names(
        catalog,
        dataset,
        1,
        {
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )


@pytest.mark.parametrize("cloud_type,version_aware", [("file", False)], indirect=True)
@pytest.mark.xdist_group(name="tmpfile")
def test_cli_query_params_metrics(
    cloud_test_catalog_tmpfile, tmp_path, catalog_info_filepath, capsys
):
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri
    catalog.create_dataset_from_sources("animals", [src_uri], recursive=True)

    query_script = """\
    from datachain.query import DatasetQuery, metrics, param

    ds = DatasetQuery(param("name"), catalog=catalog)

    metrics.set("count", ds.count())

    ds.save("my-ds")
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)

    filepath = tmp_path / "query_script.py"
    filepath.write_text(query_script)

    query(catalog, str(filepath), params={"name": "animals"})

    latest_job = get_latest_job(catalog.metastore)
    assert latest_job

    assert latest_job.status == JobStatus.COMPLETE
    assert latest_job.params == {"name": "animals"}
    assert latest_job.metrics == {"count": 7}
