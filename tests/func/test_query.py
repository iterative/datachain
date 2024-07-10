import json
import os.path
from textwrap import dedent
from typing import Optional

import dill
import pytest

from datachain.catalog import QUERY_DATASET_PREFIX
from datachain.cli import query
from datachain.data_storage import AbstractDBMetastore, JobQueryType, JobStatus
from datachain.error import QueryScriptRunError
from tests.utils import assert_row_names


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
) -> Optional[tuple[str, str, int, int, str, str]]:
    j = metastore._jobs

    latest_jobs_query = (
        metastore._jobs_select(
            j.c.id,
            j.c.name,
            j.c.status,
            j.c.query_type,
            j.c.error_message,
            j.c.error_stack,
            j.c.metrics,
        )
        .order_by(j.c.created_at.desc())
        .limit(1)
    )
    latest_jobs = list(metastore.db.execute(latest_jobs_query))
    if len(latest_jobs) == 0:
        return None
    return latest_jobs[0]


@pytest.mark.parametrize("cloud_type,version_aware", [("file", False)], indirect=True)
def test_query_cli(cloud_test_catalog_tmpfile, tmp_path, catalog_info_filepath, capsys):
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri

    query_script = f"""\
    from datachain.query import DatasetQuery

    DatasetQuery({src_uri!r}, catalog=catalog)
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)

    filepath = tmp_path / "query_script.py"
    filepath.write_text(query_script)

    ds_name = "my-dataset"
    query(catalog, str(filepath), ds_name, columns=["name"])
    captured = capsys.readouterr()

    header, *rows = captured.out.splitlines()
    assert header.strip() == "name"
    name_rows = {row.split()[1] for row in rows}
    assert name_rows == {"cat1", "cat2", "description", "dog1", "dog2", "dog3", "dog4"}

    dataset = catalog.get_dataset(ds_name)
    assert dataset
    result_job_id = dataset.get_version(dataset.latest_version).job_id
    assert result_job_id

    latest_job = get_latest_job(catalog.metastore)
    assert latest_job

    assert str(latest_job[0]) == str(result_job_id)
    assert latest_job[1] == os.path.basename(filepath)
    assert latest_job[2] == JobStatus.COMPLETE
    assert latest_job[3] == JobQueryType.PYTHON
    assert latest_job[4] == ""
    assert latest_job[5] == ""


def test_query_cli_no_dataset_returned(
    cloud_test_catalog_tmpfile, tmp_path, catalog_info_filepath, capsys
):
    catalog = cloud_test_catalog_tmpfile.catalog

    query_script = """\
    from datachain.query import DatasetQuery

    DatasetQuery("test", catalog=catalog)

    print("test")
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)

    filepath = tmp_path / "query_script.py"
    filepath.write_text(query_script)

    with pytest.raises(
        QueryScriptRunError,
        match="Last line in a script was not an instance of DatasetQuery",
    ):
        query(catalog, str(filepath), "my-dataset", columns=["name"])

    latest_job = get_latest_job(catalog.metastore)
    assert latest_job

    assert latest_job[1] == os.path.basename(filepath)
    assert latest_job[2] == JobStatus.FAILED
    assert latest_job[3] == JobQueryType.PYTHON
    assert latest_job[4] == "Last line in a script was not an instance of DatasetQuery"
    assert latest_job[5].find("datachain.error.QueryScriptRunError")


@pytest.mark.parametrize(
    "save,save_as",
    (
        (True, None),
        (None, "my-dataset"),
        (True, "my-dataset"),
    ),
)
@pytest.mark.parametrize("save_dataset", (None, "new-dataset"))
def test_query(
    save,
    save_as,
    save_dataset,
    cloud_test_catalog_tmpfile,
    tmp_path,
    catalog_info_filepath,
):
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri

    query_script = f"""\
    from datachain.query import DatasetQuery
    ds = DatasetQuery({src_uri!r}, catalog=catalog)
    if {save_dataset!r}:
        ds = ds.save({save_dataset!r})
    ds
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)

    result = catalog.query(query_script, save=save, save_as=save_as)
    if save_as:
        assert result.dataset.name == save_as
        assert catalog.get_dataset(save_as)
    elif save_dataset:
        assert result.dataset.name == save_dataset
        assert catalog.get_dataset(save_dataset)
    else:
        assert result.dataset.name.startswith(QUERY_DATASET_PREFIX)
    assert result.version == 1
    assert result.dataset.versions_values == [1]
    assert result.dataset.query_script == query_script
    assert_row_names(
        catalog,
        result.dataset,
        result.version,
        {
            "cat1",
            "cat2",
            "description",
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )


@pytest.mark.parametrize(
    "params,count",
    (
        (None, 7),
        ({"limit": 1}, 1),
        ({"limit": 5}, 5),
    ),
)
def test_query_params(
    params, count, cloud_test_catalog_tmpfile, tmp_path, catalog_info_filepath
):
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri

    query_script = f"""\
    from datachain.query import DatasetQuery, param

    ds = DatasetQuery({src_uri!r}, catalog=catalog)
    if param("limit"):
        ds = ds.limit(int(param("limit")))
    ds
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)

    result = catalog.query(query_script, save=True, params=params)
    assert (
        len(list(catalog.ls_dataset_rows(result.dataset.name, result.version))) == count
    )


def test_query_where_last_command_is_call_on_save_which_returns_attached_dataset(
    cloud_test_catalog_tmpfile, tmp_path, catalog_info_filepath
):
    """
    Testing use case where last command is call on DatasetQuery save which returns
    attached instance to underlying saved dataset
    """
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri

    query_script = f"""\
    from datachain.query import C, DatasetQuery

    DatasetQuery({src_uri!r}, catalog=catalog).filter(C.name.glob("dog*")).save("dogs")
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)

    result = catalog.query(query_script, save=True)
    assert not result.dataset.name.startswith(QUERY_DATASET_PREFIX)
    assert result.dataset.query_script == query_script
    assert result.version == 1
    assert result.dataset.versions_values == [1]
    assert_row_names(
        catalog,
        result.dataset,
        result.version,
        {
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )


def test_query_where_last_command_is_attached_dataset_query_created_from_save(
    cloud_test_catalog_tmpfile, tmp_path, catalog_info_filepath
):
    """
    Testing use case where last command is instance of DatasetQuery which is
    attached to underlying dataset by calling save just before
    """
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri

    query_script = f"""\
    from datachain.query import C, DatasetQuery

    ds = DatasetQuery(
        {src_uri!r}, catalog=catalog
    ).filter(C.name.glob("dog*")).save("dogs")
    ds
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)

    result = catalog.query(query_script, save=True)
    assert result.dataset.name == "dogs"
    assert result.dataset.query_script == query_script
    assert result.version == 1
    assert result.dataset.versions_values == [1]
    assert_row_names(
        catalog,
        result.dataset,
        result.version,
        {
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )


def test_query_where_last_command_is_attached_dataset_query_created_from_query(
    cloud_test_catalog_tmpfile, tmp_path, catalog_info_filepath
):
    """
    Testing use case where last command is instance of DatasetQuery which is
    attached to underlying dataset by creating query pointing to it
    """
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri

    query_script = f"""\
    from datachain.query import C, DatasetQuery

    ds = DatasetQuery(
        {src_uri!r}, catalog=catalog
    ).filter(C.name.glob("dog*")).save("dogs")
    DatasetQuery(name="dogs", version=1, catalog=catalog)
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)

    result = catalog.query(query_script, save=True)
    assert result.dataset.name == "dogs"
    assert result.dataset.query_script == query_script
    assert result.version == 1
    assert result.dataset.versions_values == [1]
    assert_row_names(
        catalog,
        result.dataset,
        result.version,
        {
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )


@pytest.mark.parametrize("cloud_type,version_aware", [("file", False)], indirect=True)
def test_query_params_metrics(
    cloud_test_catalog_tmpfile, tmp_path, catalog_info_filepath, capsys
):
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri

    query_script = """\
    from datachain.query import DatasetQuery, metrics, param

    ds = DatasetQuery(param("url"), catalog=catalog)

    metrics.set("count", ds.count())

    ds
    """
    query_script = setup_catalog(query_script, catalog_info_filepath)

    filepath = tmp_path / "query_script.py"
    filepath.write_text(query_script)

    query(catalog, str(filepath), params={"url": src_uri})

    latest_job = get_latest_job(catalog.metastore)
    assert latest_job

    assert latest_job[2] == JobStatus.COMPLETE
    assert json.loads(latest_job[6]) == {"count": 7}
