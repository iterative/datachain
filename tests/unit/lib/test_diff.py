import pytest
from pydantic import BaseModel

from datachain.lib.dc import DataChain
from datachain.lib.file import File
from datachain.sql.types import Int64, String
from tests.utils import sorted_dicts


@pytest.mark.parametrize("added", (True, False))
@pytest.mark.parametrize("deleted", (True, False))
@pytest.mark.parametrize("modified", (True, False))
@pytest.mark.parametrize("same", (True, False))
@pytest.mark.parametrize("status_col", ("diff", None))
@pytest.mark.parametrize("save", (True, False))
def test_compare(test_session, added, deleted, modified, same, status_col, save):
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

    if not any([added, deleted, modified, same]):
        with pytest.raises(ValueError) as exc_info:
            diff = ds1.compare(
                ds2,
                added=added,
                deleted=deleted,
                modified=modified,
                same=same,
                on=["id"],
                status_col=status_col,
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

    if save:
        diff.save("diff")
        diff = DataChain.from_dataset("diff")

    expected = []
    if modified:
        expected.append(("M", 1, "John1"))
    if added:
        expected.append(("A", 2, "Doe"))
    if deleted:
        expected.append(("D", 3, "Mark"))
    if same:
        expected.append(("S", 4, "Andy"))

    collect_fields = ["diff", "id", "name"]
    if not status_col:
        expected = [row[1:] for row in expected]
        collect_fields = collect_fields[1:]

    assert list(diff.order_by("id").collect(*collect_fields)) == expected


def test_compare_with_from_dataset(test_session):
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
        ("M", 1, "John1"),
        ("A", 2, "Doe"),
        ("D", 3, "Mark"),
        ("S", 4, "Andy"),
    ]


@pytest.mark.parametrize("added", (True, False))
@pytest.mark.parametrize("deleted", (True, False))
@pytest.mark.parametrize("modified", (True, False))
@pytest.mark.parametrize("same", (True, False))
@pytest.mark.parametrize("right_name", ("other_name", "name"))
def test_compare_with_explicit_compare_fields(
    test_session, added, deleted, modified, same, right_name
):
    if not any([added, deleted, modified, same]):
        pytest.skip("This case is tested in another test")

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
        added=added,
        deleted=deleted,
        modified=modified,
        same=same,
        status_col="diff",
    )

    string_default = String.default_value(test_session.catalog.warehouse.db.dialect)

    expected = []
    if modified:
        expected.append(("M", 1, "John1", "New York"))
    if added:
        expected.append(("A", 2, "Doe", "Boston"))
    if deleted:
        expected.append(
            (
                "D",
                3,
                string_default if right_name == "other_name" else "Mark",
                "Seattle",
            )
        )
    if same:
        expected.append(("S", 4, "Andy", "San Francisco"))

    collect_fields = ["diff", "id", "name", "city"]
    assert list(diff.order_by("id").collect(*collect_fields)) == expected


@pytest.mark.parametrize("added", (True, False))
@pytest.mark.parametrize("deleted", (True, False))
@pytest.mark.parametrize("modified", (True, False))
@pytest.mark.parametrize("same", (True, False))
def test_compare_different_left_right_on_columns(
    test_session, added, deleted, modified, same
):
    if not any([added, deleted, modified, same]):
        pytest.skip("This case is tested in another test")

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
        added=added,
        deleted=deleted,
        modified=modified,
        same=same,
        on=["id"],
        right_on=["other_id"],
        status_col="diff",
    )

    int_default = Int64.default_value(test_session.catalog.warehouse.db.dialect)

    expected = []
    if same:
        expected.append(("S", 4, "Andy"))
    if added:
        expected.append(("A", 2, "Doe"))
    if modified:
        expected.append(("M", 1, "John1"))
    if deleted:
        expected.append(("D", int_default, "Mark"))

    collect_fields = ["diff", "id", "name"]
    assert list(diff.order_by("name").collect(*collect_fields)) == expected


@pytest.mark.parametrize("added", (True, False))
@pytest.mark.parametrize("deleted", (True, False))
@pytest.mark.parametrize("modified", (True, False))
@pytest.mark.parametrize("same", (True, False))
@pytest.mark.parametrize("on_self", (True, False))
def test_compare_on_equal_datasets(
    test_session, added, deleted, modified, same, on_self
):
    if not any([added, deleted, modified, same]):
        pytest.skip("This case is tested in another test")

    ds1 = DataChain.from_values(
        id=[1, 2, 3],
        name=["John", "Doe", "Andy"],
        session=test_session,
    ).save("ds1")

    if on_self:
        ds2 = ds1
    else:
        ds2 = DataChain.from_values(
            id=[1, 2, 3],
            name=["John", "Doe", "Andy"],
            session=test_session,
        ).save("ds2")

    diff = ds1.compare(
        ds2,
        added=added,
        deleted=deleted,
        modified=modified,
        same=same,
        on=["id"],
        status_col="diff",
    )

    if not same:
        expected = []
    else:
        expected = [
            ("S", 1, "John"),
            ("S", 2, "Doe"),
            ("S", 3, "Andy"),
        ]

    collect_fields = ["diff", "id", "name"]
    assert list(diff.order_by("id").collect(*collect_fields)) == expected


def test_compare_multiple_columns(test_session):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John", "Doe", "Andy"],
        city=["London", "New York", "Tokyo"],
        session=test_session,
    ).save("ds1")
    ds2 = DataChain.from_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        city=["Paris", "Berlin", "Tokyo"],
        session=test_session,
    ).save("ds2")

    diff = ds1.compare(ds2, same=True, on=["id"], status_col="diff")

    assert sorted_dicts(diff.to_records(), "id") == sorted_dicts(
        [
            {"diff": "M", "id": 1, "name": "John", "city": "London"},
            {"diff": "A", "id": 2, "name": "Doe", "city": "New York"},
            {"diff": "D", "id": 3, "name": "Mark", "city": "Berlin"},
            {"diff": "S", "id": 4, "name": "Andy", "city": "Tokyo"},
        ],
        "id",
    )


def test_compare_multiple_match_columns(test_session):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John", "Doe", "Andy"],
        city=["London", "New York", "Tokyo"],
        session=test_session,
    ).save("ds1")
    ds2 = DataChain.from_values(
        id=[1, 3, 4],
        name=["John", "John", "Andy"],
        city=["Paris", "Berlin", "Tokyo"],
        session=test_session,
    ).save("ds2")

    diff = ds1.compare(ds2, same=True, on=["id", "name"], status_col="diff")

    assert sorted_dicts(diff.to_records(), "id") == sorted_dicts(
        [
            {"diff": "M", "id": 1, "name": "John", "city": "London"},
            {"diff": "A", "id": 2, "name": "Doe", "city": "New York"},
            {"diff": "D", "id": 3, "name": "John", "city": "Berlin"},
            {"diff": "S", "id": 4, "name": "Andy", "city": "Tokyo"},
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
            {"diff": "M", "id": 1, "name": "John", "city": "London"},
            {"diff": "A", "id": 2, "name": "Doe", "city": "New York"},
            {"diff": "D", "id": 3, "name": "Mark", "city": string_default},
            {"diff": "M", "id": 4, "name": "Andy", "city": "Tokyo"},
        ],
        "id",
    )


def test_compare_additional_column_on_right(test_session):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John", "Doe", "Andy"],
        session=test_session,
    ).save("ds1")
    ds2 = DataChain.from_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        city=["London", "New York", "Tokyo"],
        session=test_session,
    ).save("ds2")

    diff = ds1.compare(ds2, same=True, on=["id"], status_col="diff")

    assert sorted_dicts(diff.to_records(), "id") == sorted_dicts(
        [
            {"diff": "M", "id": 1, "name": "John"},
            {"diff": "A", "id": 2, "name": "Doe"},
            {"diff": "D", "id": 3, "name": "Mark"},
            {"diff": "M", "id": 4, "name": "Andy"},
        ],
        "id",
    )


def test_compare_missing_on(test_session):
    ds1 = DataChain.from_values(id=[1, 2, 4], session=test_session).save("ds1")
    ds2 = DataChain.from_values(id=[1, 2, 4], session=test_session).save("ds2")

    with pytest.raises(ValueError) as exc_info:
        ds1.compare(ds2, on=None)

    assert str(exc_info.value) == "'on' must be specified"


def test_compare_right_on_wrong_length(test_session):
    ds1 = DataChain.from_values(id=[1, 2, 4], session=test_session).save("ds1")
    ds2 = DataChain.from_values(id=[1, 2, 4], session=test_session).save("ds2")

    with pytest.raises(ValueError) as exc_info:
        ds1.compare(ds2, on=["id"], right_on=["id", "name"])

    assert str(exc_info.value) == "'on' and 'right_on' must be have the same length"


def test_compare_right_compare_defined_but_not_compare(test_session):
    ds1 = DataChain.from_values(id=[1, 2, 4], session=test_session).save("ds1")
    ds2 = DataChain.from_values(id=[1, 2, 4], session=test_session).save("ds2")

    with pytest.raises(ValueError) as exc_info:
        ds1.compare(ds2, on=["id"], right_compare=["name"])

    assert str(exc_info.value) == (
        "'compare' must be defined if 'right_compare' is defined"
    )


def test_compare_right_compare_wrong_length(test_session):
    ds1 = DataChain.from_values(id=[1, 2, 4], session=test_session).save("ds1")
    ds2 = DataChain.from_values(id=[1, 2, 4], session=test_session).save("ds2")

    with pytest.raises(ValueError) as exc_info:
        ds1.compare(ds2, on=["id"], compare=["name"], right_compare=["name", "city"])

    assert str(exc_info.value) == (
        "'compare' and 'right_compare' must be have the same length"
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
        ("M", fs1_updated, 1),
        ("A", fs2, 2),
        ("D", fs3, 3),
        ("S", fs4, 4),
    ]

    collect_fields = ["diff", "file", "score"]
    if not status_col:
        expected = [row[1:] for row in expected]
        collect_fields = collect_fields[1:]

    assert list(diff.order_by("file.source").collect(*collect_fields)) == expected


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
        ("M", fs1_updated, 1),
        ("A", fs2, 2),
        ("D", fs3, 3),
        ("S", fs4, 4),
    ]

    collect_fields = ["diff", "nested", "score"]
    if not status_col:
        expected = [row[1:] for row in expected]
        collect_fields = collect_fields[1:]

    assert (
        list(diff.order_by("nested.file.source").collect(*collect_fields)) == expected
    )
