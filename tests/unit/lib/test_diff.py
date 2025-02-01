from datetime import timezone

import pytest
from pydantic import BaseModel

from datachain.diff import CompareStatus, compare_and_split
from datachain.lib.dc import DataChain
from datachain.lib.file import File
from datachain.sql.types import Int64, String
from tests.utils import sorted_dicts


def _as_utc(d):
    return d.replace(tzinfo=timezone.utc)


@pytest.mark.parametrize("added", (True, False))
@pytest.mark.parametrize("deleted", (True, False))
@pytest.mark.parametrize("modified", (True, False))
@pytest.mark.parametrize("same", (True, False))
def test_compare(test_session, added, deleted, modified, same):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John1", "Doe", "Andy"],
        session=test_session,
    )

    ds2 = DataChain.from_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        session=test_session,
    )

    if not any([added, deleted, modified, same]):
        with pytest.raises(ValueError) as exc_info:
            diff = ds1.compare(
                ds2,
                added=added,
                deleted=deleted,
                modified=modified,
                same=same,
                on=["id"],
                status_col="diff",
            )
        assert str(exc_info.value) == (
            "At least one of added, deleted, modified, same flags must be set"
        )
        return

    diff = ds1.compare(
        ds2,
        added=added,
        deleted=deleted,
        modified=modified,
        same=same,
        on=["id"],
        status_col="diff",
    )

    chains = compare_and_split(
        ds1,
        ds2,
        same=True,
        on=["id"],
    )

    expected = []

    if modified:
        assert "diff" not in chains[CompareStatus.MODIFIED].signals_schema.db_signals()
        expected.append((CompareStatus.MODIFIED, 1, "John1"))

    if added:
        assert "diff" not in chains[CompareStatus.ADDED].signals_schema.db_signals()
        expected.append((CompareStatus.ADDED, 2, "Doe"))

    if deleted:
        assert "diff" not in chains[CompareStatus.DELETED].signals_schema.db_signals()
        expected.append((CompareStatus.DELETED, 3, "Mark"))

    if same:
        assert "diff" not in chains[CompareStatus.SAME].signals_schema.db_signals()
        expected.append((CompareStatus.SAME, 4, "Andy"))

    assert list(diff.order_by("id").collect("diff", "id", "name")) == expected


def test_compare_no_status_col(test_session):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John1", "Doe", "Andy"],
        session=test_session,
    )

    ds2 = DataChain.from_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        session=test_session,
    )

    diff = ds1.compare(
        ds2,
        same=True,
        on=["id"],
        status_col=None,
    )

    expected = [
        (1, "John1"),
        (2, "Doe"),
        (3, "Mark"),
        (4, "Andy"),
    ]

    assert list(diff.order_by("id").collect()) == expected


def test_compare_from_datasets(test_session):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John1", "Doe", "Andy"],
        session=test_session,
    ).save("ds1")

    ds2 = DataChain.from_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        session=test_session,
    ).save("ds2")

    # this adds sys columns to ds1 and ds2
    ds1 = DataChain.from_dataset("ds1")
    ds2 = DataChain.from_dataset("ds2")

    diff = ds1.compare(ds2, same=True, on=["id"], status_col="diff")

    assert list(diff.order_by("id").collect("diff", "id", "name")) == [
        (CompareStatus.MODIFIED, 1, "John1"),
        (CompareStatus.ADDED, 2, "Doe"),
        (CompareStatus.DELETED, 3, "Mark"),
        (CompareStatus.SAME, 4, "Andy"),
    ]


@pytest.mark.parametrize("right_name", ("other_name", "name"))
def test_compare_with_explicit_compare_fields(test_session, right_name):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John1", "Doe", "Andy"],
        city=["New York", "Boston", "San Francisco"],
        session=test_session,
    ).save("ds1")

    ds2_data = {
        "id": [1, 3, 4],
        "city": ["Washington", "Seattle", "Miami"],
        f"{right_name}": ["John", "Mark", "Andy"],
        "session": test_session,
    }

    ds2 = DataChain.from_values(**ds2_data).save("ds2")

    diff = ds1.compare(
        ds2,
        on=["id"],
        compare=["name"],
        right_compare=[right_name],
        same=True,
        status_col="diff",
    )

    string_default = String.default_value(test_session.catalog.warehouse.db.dialect)

    expected = [
        (CompareStatus.MODIFIED, 1, "John1", "New York"),
        (CompareStatus.ADDED, 2, "Doe", "Boston"),
        (
            CompareStatus.DELETED,
            3,
            string_default if right_name == "other_name" else "Mark",
            "Seattle",
        ),
        (CompareStatus.SAME, 4, "Andy", "San Francisco"),
    ]

    collect_fields = ["diff", "id", "name", "city"]
    assert list(diff.order_by("id").collect(*collect_fields)) == expected


def test_compare_different_left_right_on_columns(test_session):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John1", "Doe", "Andy"],
        session=test_session,
    ).save("ds1")

    ds2 = DataChain.from_values(
        other_id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        session=test_session,
    ).save("ds2")

    diff = ds1.compare(
        ds2,
        same=True,
        on=["id"],
        right_on=["other_id"],
        status_col="diff",
    )

    int_default = Int64.default_value(test_session.catalog.warehouse.db.dialect)

    expected = [
        (CompareStatus.SAME, 4, "Andy"),
        (CompareStatus.ADDED, 2, "Doe"),
        (CompareStatus.MODIFIED, 1, "John1"),
        (CompareStatus.DELETED, int_default, "Mark"),
    ]

    collect_fields = ["diff", "id", "name"]
    assert list(diff.order_by("name").collect(*collect_fields)) == expected


@pytest.mark.parametrize("on_self", (True, False))
def test_compare_on_equal_datasets(test_session, on_self):
    ds1 = DataChain.from_values(
        id=[1, 2, 3],
        name=["John", "Doe", "Andy"],
        session=test_session,
    )

    if on_self:
        ds2 = ds1
    else:
        ds2 = DataChain.from_values(
            id=[1, 2, 3],
            name=["John", "Doe", "Andy"],
            session=test_session,
        )

    diff = ds1.compare(
        ds2,
        same=True,
        on=["id"],
        status_col="diff",
    )

    expected = [
        (CompareStatus.SAME, 1, "John"),
        (CompareStatus.SAME, 2, "Doe"),
        (CompareStatus.SAME, 3, "Andy"),
    ]

    collect_fields = ["diff", "id", "name"]
    assert list(diff.order_by("id").collect(*collect_fields)) == expected


def test_compare_multiple_columns(test_session):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John", "Doe", "Andy"],
        city=["London", "New York", "Tokyo"],
        session=test_session,
    )
    ds2 = DataChain.from_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        city=["Paris", "Berlin", "Tokyo"],
        session=test_session,
    )

    diff = ds1.compare(ds2, same=True, on=["id"], status_col="diff")

    assert sorted_dicts(diff.to_records(), "id") == sorted_dicts(
        [
            {"diff": CompareStatus.MODIFIED, "id": 1, "name": "John", "city": "London"},
            {"diff": CompareStatus.ADDED, "id": 2, "name": "Doe", "city": "New York"},
            {"diff": CompareStatus.DELETED, "id": 3, "name": "Mark", "city": "Berlin"},
            {"diff": CompareStatus.SAME, "id": 4, "name": "Andy", "city": "Tokyo"},
        ],
        "id",
    )


def test_compare_multiple_match_columns(test_session):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John", "Doe", "Andy"],
        city=["London", "New York", "Tokyo"],
        session=test_session,
    )
    ds2 = DataChain.from_values(
        id=[1, 3, 4],
        name=["John", "John", "Andy"],
        city=["Paris", "Berlin", "Tokyo"],
        session=test_session,
    )

    diff = ds1.compare(ds2, same=True, on=["id", "name"], status_col="diff")

    assert sorted_dicts(diff.to_records(), "id") == sorted_dicts(
        [
            {"diff": CompareStatus.MODIFIED, "id": 1, "name": "John", "city": "London"},
            {"diff": CompareStatus.ADDED, "id": 2, "name": "Doe", "city": "New York"},
            {"diff": CompareStatus.DELETED, "id": 3, "name": "John", "city": "Berlin"},
            {"diff": CompareStatus.SAME, "id": 4, "name": "Andy", "city": "Tokyo"},
        ],
        "id",
    )


def test_compare_additional_column_on_left(test_session):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John", "Doe", "Andy"],
        city=["London", "New York", "Tokyo"],
        session=test_session,
    ).save("ds1")
    ds2 = DataChain.from_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        session=test_session,
    ).save("ds2")

    string_default = String.default_value(test_session.catalog.warehouse.db.dialect)

    diff = ds1.compare(ds2, same=True, on=["id"], status_col="diff")

    assert sorted_dicts(diff.to_records(), "id") == sorted_dicts(
        [
            {"diff": CompareStatus.MODIFIED, "id": 1, "name": "John", "city": "London"},
            {"diff": CompareStatus.ADDED, "id": 2, "name": "Doe", "city": "New York"},
            {
                "diff": CompareStatus.DELETED,
                "id": 3,
                "name": "Mark",
                "city": string_default,
            },
            {"diff": CompareStatus.MODIFIED, "id": 4, "name": "Andy", "city": "Tokyo"},
        ],
        "id",
    )


def test_compare_additional_column_on_right(test_session):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John", "Doe", "Andy"],
        session=test_session,
    )
    ds2 = DataChain.from_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        city=["London", "New York", "Tokyo"],
        session=test_session,
    )

    diff = ds1.compare(ds2, same=True, on=["id"], status_col="diff")

    assert sorted_dicts(diff.to_records(), "id") == sorted_dicts(
        [
            {"diff": CompareStatus.MODIFIED, "id": 1, "name": "John"},
            {"diff": CompareStatus.ADDED, "id": 2, "name": "Doe"},
            {"diff": CompareStatus.DELETED, "id": 3, "name": "Mark"},
            {"diff": CompareStatus.MODIFIED, "id": 4, "name": "Andy"},
        ],
        "id",
    )


def test_compare_missing_on(test_session):
    ds1 = DataChain.from_values(id=[1, 2, 4], session=test_session)
    ds2 = DataChain.from_values(id=[1, 2, 4], session=test_session)

    with pytest.raises(ValueError) as exc_info:
        ds1.compare(ds2, on=None)

    assert str(exc_info.value) == "'on' must be specified"


def test_compare_right_on_wrong_length(test_session):
    ds1 = DataChain.from_values(id=[1, 2, 4], session=test_session)
    ds2 = DataChain.from_values(id=[1, 2, 4], session=test_session)

    with pytest.raises(ValueError) as exc_info:
        ds1.compare(ds2, on=["id"], right_on=["id", "name"])

    assert str(exc_info.value) == "'on' and 'right_on' must be have the same length"


def test_compare_right_compare_defined_but_not_compare(test_session):
    ds1 = DataChain.from_values(id=[1, 2, 4], session=test_session)
    ds2 = DataChain.from_values(id=[1, 2, 4], session=test_session)

    with pytest.raises(ValueError) as exc_info:
        ds1.compare(ds2, on=["id"], right_compare=["name"])

    assert str(exc_info.value) == (
        "'compare' must be defined if 'right_compare' is defined"
    )


def test_compare_right_compare_wrong_length(test_session):
    ds1 = DataChain.from_values(id=[1, 2, 4], session=test_session)
    ds2 = DataChain.from_values(id=[1, 2, 4], session=test_session)

    with pytest.raises(ValueError) as exc_info:
        ds1.compare(ds2, on=["id"], compare=["name"], right_compare=["name", "city"])

    assert str(exc_info.value) == (
        "'compare' and 'right_compare' must have the same length"
    )


@pytest.mark.parametrize("status_col", ("diff", None))
def test_diff(test_session, status_col):
    fs1 = File(source="s1", path="p1", version="2", etag="e2")
    fs1_updated = File(source="s1", path="p1", version="1", etag="e1")
    fs2 = File(source="s2", path="p2", version="1", etag="e1")
    fs3 = File(source="s3", path="p3", version="1", etag="e1")
    fs4 = File(source="s4", path="p4", version="1", etag="e1")

    ds1 = DataChain.from_values(
        file=[fs1_updated, fs2, fs4], score=[1, 2, 4], session=test_session
    )
    ds2 = DataChain.from_values(
        file=[fs1, fs3, fs4], score=[1, 3, 4], session=test_session
    )

    diff = ds1.diff(
        ds2,
        added=True,
        deleted=True,
        modified=True,
        same=True,
        on="file",
        status_col=status_col,
    )

    expected = [
        (CompareStatus.MODIFIED, fs1_updated, 1),
        (CompareStatus.ADDED, fs2, 2),
        (CompareStatus.DELETED, fs3, 3),
        (CompareStatus.SAME, fs4, 4),
    ]

    collect_fields = ["diff", "file", "score"]
    if not status_col:
        expected = [row[1:] for row in expected]
        collect_fields = collect_fields[1:]

    res = list(diff.order_by("file.source").collect(*collect_fields))
    for r in res:
        r[-2].last_modified = _as_utc(r[-2].last_modified)

    assert res == expected


@pytest.mark.parametrize("status_col", ("diff", None))
def test_diff_nested(test_session, status_col):
    class Nested(BaseModel):
        file: File

    fs1 = Nested(file=File(source="s1", path="p1", version="2", etag="e2"))
    fs1_updated = Nested(file=File(source="s1", path="p1", version="1", etag="e1"))
    fs2 = Nested(file=File(source="s2", path="p2", version="1", etag="e1"))
    fs3 = Nested(file=File(source="s3", path="p3", version="1", etag="e1"))
    fs4 = Nested(file=File(source="s4", path="p4", version="1", etag="e1"))

    ds1 = DataChain.from_values(
        nested=[fs1_updated, fs2, fs4], score=[1, 2, 4], session=test_session
    )
    ds2 = DataChain.from_values(
        nested=[fs1, fs3, fs4], score=[1, 3, 4], session=test_session
    )

    diff = ds1.diff(
        ds2,
        added=True,
        deleted=True,
        modified=True,
        same=True,
        on="nested.file",
        status_col=status_col,
    )

    expected = [
        (CompareStatus.MODIFIED, fs1_updated, 1),
        (CompareStatus.ADDED, fs2, 2),
        (CompareStatus.DELETED, fs3, 3),
        (CompareStatus.SAME, fs4, 4),
    ]

    collect_fields = ["diff", "nested", "score"]
    if not status_col:
        expected = [row[1:] for row in expected]
        collect_fields = collect_fields[1:]

    res = list(diff.order_by("nested.file.source").collect(*collect_fields))
    for r in res:
        r[-2].file.last_modified = _as_utc(r[-2].file.last_modified)
    assert res == expected
