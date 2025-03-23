from datetime import datetime
from typing import Any

import pytest

from datachain.sql.types import (
    JSON,
    Array,
    Boolean,
    DateTime,
    Float,
    Float32,
    Float64,
    Int,
    String,
)
from tests.utils import (
    DEFAULT_TREE,
    TARRED_TREE,
    create_tar_dataset_with_legacy_columns,
)

COMPLEX_TREE: dict[str, Any] = {
    **TARRED_TREE,
    **DEFAULT_TREE,
    "nested": {"dir": {"path": {"abc.txt": "abc"}}},
}


@pytest.mark.parametrize("tree", [COMPLEX_TREE], indirect=True)
def test_dir_expansion(cloud_test_catalog, version_aware, cloud_type):
    has_version = version_aware or cloud_type == "gs"

    ctc = cloud_test_catalog
    session = ctc.session
    catalog = ctc.catalog
    src_uri = ctc.src_uri
    if cloud_type == "file":
        # we don't want to index things in parent directory
        src_uri += "/"

    dc = create_tar_dataset_with_legacy_columns(session, ctc.src_uri, "dc")
    dataset = catalog.get_dataset(dc.name)
    with catalog.warehouse.clone() as warehouse:
        dr = warehouse.dataset_rows(dataset, object_name="file")
        de = dr.dir_expansion()
        q = de.query(dr.get_table())

        columns = (
            "id",
            "is_dir",
            "source",
            "path",
            "version",
            "location",
        )

        result = [dict(zip(columns, r)) for r in warehouse.db.execute(q)]
        to_compare = [(r["path"], r["is_dir"], r["version"] != "") for r in result]

    assert all(r["source"] == ctc.src_uri for r in result)

    # Note, we have both a file and a directory entry for expanded tar files
    expected = [
        ("animals.tar", 0, has_version),
        ("animals.tar", 1, False),
        ("animals.tar/cats", 1, False),
        ("animals.tar/cats/cat1", 0, has_version),
        ("animals.tar/cats/cat2", 0, has_version),
        ("animals.tar/description", 0, has_version),
        ("animals.tar/dogs", 1, False),
        ("animals.tar/dogs/dog1", 0, has_version),
        ("animals.tar/dogs/dog2", 0, has_version),
        ("animals.tar/dogs/dog3", 0, has_version),
        ("animals.tar/dogs/others", 1, False),
        ("animals.tar/dogs/others/dog4", 0, has_version),
        ("cats", 1, False),
        ("cats/cat1", 0, has_version),
        ("cats/cat2", 0, has_version),
        ("description", 0, has_version),
        ("dogs", 1, False),
        ("dogs/dog1", 0, has_version),
        ("dogs/dog2", 0, has_version),
        ("dogs/dog3", 0, has_version),
        ("dogs/others", 1, False),
        ("dogs/others/dog4", 0, has_version),
        ("nested", 1, False),
        ("nested/dir", 1, False),
        ("nested/dir/path", 1, False),
        ("nested/dir/path/abc.txt", 0, has_version),
    ]

    assert to_compare == expected


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_convert_type(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    warehouse = catalog.warehouse
    now = datetime.now()

    def run_convert_type(value, sql_type):
        return warehouse.convert_type(
            value,
            sql_type,
            warehouse.python_type(sql_type),
            type(sql_type).__name__,
            "test_column",
        )

    # convert int to float
    for f in [Float, Float32, Float64]:
        converted = run_convert_type(1, f())
        assert converted == 1.0
        assert isinstance(converted, float)

    # types match, nothing to convert
    assert run_convert_type(1, Int()) == 1
    assert run_convert_type(1.5, Float()) == 1.5
    assert run_convert_type(True, Boolean()) is True
    assert run_convert_type("s", String()) == "s"
    assert run_convert_type(now, DateTime()) == now
    assert run_convert_type([1, 2], Array(Int)) == [1, 2]
    assert run_convert_type([1.5, 2.5], Array(Float)) == [1.5, 2.5]
    assert run_convert_type(["a", "b"], Array(String)) == ["a", "b"]
    assert run_convert_type([[1, 2], [3, 4]], Array(Array(Int))) == [
        [1, 2],
        [3, 4],
    ]

    # JSON Tests
    assert run_convert_type('{"a": 1}', JSON()) == '{"a": 1}'
    assert run_convert_type({"a": 1}, JSON()) == '{"a": 1}'
    assert run_convert_type([{"a": 1}], JSON()) == '[{"a": 1}]'
    with pytest.raises(ValueError):
        run_convert_type(0.5, JSON())

    # convert array to compatible type
    converted = run_convert_type([1, 2], Array(Float))
    assert converted == [1.0, 2.0]
    assert all(isinstance(c, float) for c in converted)

    # convert nested array to compatible type
    converted = run_convert_type([[1, 2], [3, 4]], Array(Array(Float)))
    assert converted == [[1.0, 2.0], [3.0, 4.0]]
    assert all(isinstance(c, float) for c in converted[0])
    assert all(isinstance(c, float) for c in converted[1])

    # error, float to int
    with pytest.raises(ValueError):
        run_convert_type(1.5, Int())

    # error, float to int in list
    with pytest.raises(ValueError):
        run_convert_type([1.5, 1], Array(Int))
