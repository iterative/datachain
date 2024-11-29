import math
from typing import Optional

import pandas as pd
import pytest
from pydantic import BaseModel
from sqlalchemy import func

from datachain.lib.dc import C, DataChain, DatasetMergeError
from datachain.sql.types import Int, String
from tests.utils import skip_if_not_sqlite


class User(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None


class Player(User):
    weight: Optional[float] = None
    height: Optional[int] = None


class Employee(BaseModel):
    id: Optional[int] = None
    person: User


class TeamMember(BaseModel):
    player: Optional[str] = None
    sport: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None


employees = [
    Employee(id=151, person=User(name="Alice", age=31)),
    Employee(id=152, person=User(name="Bob", age=27)),
    Employee(id=153, person=User(name="Charlie", age=54)),
    Employee(id=154, person=User(name="David", age=29)),
]
team = [
    TeamMember(player="Alice", sport="volleyball", weight=120.3, height=5.5),
    TeamMember(player="Charlie", sport="football", weight=200.0, height=6.0),
    TeamMember(player="David", sport="football", weight=158.7, height=5.7),
]


def test_merge_objects(test_session):
    ch1 = DataChain.from_values(emp=employees, session=test_session)
    ch2 = DataChain.from_values(team=team, session=test_session)
    ch = ch1.merge(ch2, "emp.person.name", "team.player")

    str_default = String.default_value(test_session.catalog.warehouse.db.dialect)

    i = 0
    j = 0
    for items in ch.order_by("emp.person.name", "team.player").collect():
        assert len(items) == 2

        empl, player = items
        assert isinstance(empl, Employee)
        assert empl == employees[i]
        i += 1

        assert isinstance(player, TeamMember)
        if empl.person.name != "Bob":
            assert player.player == team[j].player
            assert player.sport == team[j].sport
            assert math.isclose(player.weight, team[j].weight, rel_tol=1e-7)
            assert math.isclose(player.height, team[j].height, rel_tol=1e-7)
            j += 1
        else:
            assert player.player == str_default
            assert player.sport == str_default
            assert pd.isnull(player.weight)
            assert pd.isnull(player.height)

    assert i == len(employees)
    assert j == len(team)


def test_merge_similar_objects(test_session):
    new_employees = [
        Employee(id=152, person=User(name="Bob", age=27)),
        Employee(id=201, person=User(name="Karl", age=18)),
        Employee(id=154, person=User(name="David", age=29)),
    ]

    ch1 = DataChain.from_values(emp=employees, session=test_session)
    ch2 = DataChain.from_values(emp=new_employees, session=test_session)

    rname = "qq"
    ch = ch1.merge(ch2, "emp.person.name", rname=rname)

    assert list(ch.signals_schema.values.keys()) == ["sys", "emp", rname + "emp"]

    empl = list(ch.collect())
    assert len(empl) == 4
    assert len(empl[0]) == 2

    ch_inner = ch1.merge(ch2, "emp.person.name", rname=rname, inner=True)
    assert len(list(ch_inner.collect())) == 2


@skip_if_not_sqlite
def test_merge_similar_objects_in_memory():
    # Skip if not on SQLite, as in_memory databases are only supported on SQLite
    new_employees = [
        Employee(id=152, person=User(name="Bob", age=27)),
        Employee(id=201, person=User(name="Karl", age=18)),
        Employee(id=154, person=User(name="David", age=29)),
    ]

    ch1 = DataChain.from_values(emp=employees, in_memory=True)
    # This should use the same session as above (in_memory=True automatically)
    ch2 = DataChain.from_values(emp=new_employees)
    assert ch1.session.catalog.in_memory is True
    assert ch1.session.catalog.metastore.db.db_file == ":memory:"
    assert ch1.session.catalog.warehouse.db.db_file == ":memory:"
    assert ch2.session.catalog.in_memory is True
    assert ch2.session.catalog.metastore.db.db_file == ":memory:"
    assert ch2.session.catalog.warehouse.db.db_file == ":memory:"

    rname = "qq"
    ch = ch1.merge(ch2, "emp.person.name", rname=rname)
    assert ch.session.catalog.in_memory is True
    assert ch.session.catalog.metastore.db.db_file == ":memory:"
    assert ch.session.catalog.warehouse.db.db_file == ":memory:"

    assert list(ch.signals_schema.values.keys()) == ["sys", "emp", rname + "emp"]

    empl = list(ch.collect())
    assert len(empl) == 4
    assert len(empl[0]) == 2

    ch_inner = ch1.merge(ch2, "emp.person.name", rname=rname, inner=True)
    assert len(list(ch_inner.collect())) == 2


def test_merge_values(test_session):
    order_ids = [11, 22, 33, 44]
    order_descr = ["water", "water", "paper", "water"]

    delivery_ids = [11, 44]
    delivery_time = [24.0, 16.5]

    ch1 = DataChain.from_values(id=order_ids, descr=order_descr, session=test_session)
    ch2 = DataChain.from_values(
        id=delivery_ids, time=delivery_time, session=test_session
    )

    ch = ch1.merge(ch2, "id")

    assert list(ch.signals_schema.values.keys()) == [
        "sys",
        "id",
        "descr",
        "right_id",
        "time",
    ]

    i = 0
    j = 0
    sorted_items_list = sorted(ch.collect(), key=lambda x: x[0])
    for items in sorted_items_list:
        assert len(items) == 4
        id, name, _right_id, time = items

        assert id == order_ids[i]
        assert name == order_descr[i]
        i += 1

        if pd.notnull(time):
            assert id == delivery_ids[j]
            assert time == delivery_time[j]
            j += 1

    assert i == len(order_ids)
    assert j == len(delivery_ids)


def test_merge_multi_conditions(test_session):
    order_ids = [11, 22, 33, 44]
    order_name = ["water", "water", "paper", "water"]
    order_descr = ["still water", "still water", "white paper", "sparkling water"]

    delivery_ids = [11, 44]
    delivery_name = ["water", "unknown"]
    delivery_time = [24.0, 16.5]

    ch1 = DataChain.from_values(
        id=order_ids, name=order_name, descr=order_descr, session=test_session
    )
    ch2 = DataChain.from_values(
        id=delivery_ids, d_name=delivery_name, time=delivery_time, session=test_session
    )

    ch = ch1.merge(ch2, ("id", "name"), ("id", C("d_name")))

    res = list(ch.collect())

    assert len(res) == max(len(employees), len(team))
    success_ids = set()
    for items in res:
        if items[3]:
            success_ids.add(items[0])

    assert success_ids == {11}


def test_merge_errors(test_session):
    ch1 = DataChain.from_values(emp=employees, session=test_session)
    ch2 = DataChain.from_values(team=team, session=test_session)

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, "unknown")

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, ["emp.person.name"], "unknown")

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, ["emp.person.name"], ["unknown"])

    with pytest.raises(DatasetMergeError):
        ch1.merge(
            ch2, ("emp.person.age", func.substr(["emp.person.name"], 2)), "unknown"
        )

    ch1.merge(ch2, ["emp.person.name"], ["team.sport"])
    ch1.merge(ch2, ["emp.person.name"], ["team.sport"])

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, ["emp.person.name"], ["team.player", "team.sport"])

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, 33)

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, "emp.person.name", True)


def test_merge_with_itself(test_session):
    ch = DataChain.from_values(emp=employees, session=test_session)
    merged = ch.merge(ch, "emp.id")

    count = 0
    for left, right in merged.order_by("emp.id").collect():
        assert isinstance(left, Employee)
        assert isinstance(right, Employee)
        assert left == right == employees[count]
        count += 1

    assert count == len(employees)


def test_merge_with_itself_column(test_session):
    ch = DataChain.from_values(emp=employees, session=test_session)
    merged = ch.merge(ch, C("emp.id"))

    count = 0
    for left, right in merged.order_by("emp.id").collect():
        assert isinstance(left, Employee)
        assert isinstance(right, Employee)
        assert left == right == employees[count]
        count += 1

    assert count == len(employees)


def test_merge_on_expression(test_session):
    def _get_expr(dc):
        c = dc.c("team.sport")
        return func.substr(c, func.length(c) - 3)

    dc = DataChain.from_values(team=team, session=test_session)
    right_dc = dc.clone()

    # cross join on "ball" from sport
    merged = dc.merge(right_dc, on=_get_expr(dc), right_on=_get_expr(right_dc))

    cross_team = [
        (left_member, right_member) for left_member in team for right_member in team
    ]

    merged.show()
    count = 0
    for left, right_dc in merged.order_by("team.player", "right_team.player").collect():
        assert isinstance(left, TeamMember)
        assert isinstance(right_dc, TeamMember)
        left_member, right_member = cross_team[count]
        assert left == left_member
        assert right_dc == right_member
        count += 1

    assert count == len(team) * len(team)


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
def test_merge_union(cloud_test_catalog, inner, cloud_type):
    catalog = cloud_test_catalog.catalog
    session = cloud_test_catalog.session

    src = cloud_test_catalog.src_uri

    dogs = DataChain.from_storage(f"{src}/dogs/*", session=session)
    cats = DataChain.from_storage(f"{src}/cats/*", session=session)

    signal_default_value = Int.default_value(catalog.warehouse.db.dialect)

    dogs1 = dogs.map(sig1=lambda: 1, output={"sig1": int})
    dogs2 = dogs.map(sig2=lambda: 2, output={"sig2": int})
    cats1 = cats.map(sig1=lambda: 1, output={"sig1": int})

    merged = (dogs1 | cats1).merge(dogs2, "file.path", inner=inner)
    signals = merged.select("file.path", "sig1", "sig2").order_by("file.path").results()

    if inner:
        assert signals == [
            ("dogs/dog1", 1, 2),
            ("dogs/dog2", 1, 2),
            ("dogs/dog3", 1, 2),
            ("dogs/others/dog4", 1, 2),
        ]
    else:
        assert signals == [
            ("cats/cat1", 1, signal_default_value),
            ("cats/cat2", 1, signal_default_value),
            ("dogs/dog1", 1, 2),
            ("dogs/dog2", 1, 2),
            ("dogs/dog3", 1, 2),
            ("dogs/others/dog4", 1, 2),
        ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner1", [True, False])
@pytest.mark.parametrize("inner2", [True, False])
@pytest.mark.parametrize("inner3", [True, False])
def test_merge_multiple(cloud_test_catalog, inner1, inner2, inner3):
    catalog = cloud_test_catalog.catalog
    session = cloud_test_catalog.session

    src = cloud_test_catalog.src_uri

    dogs = DataChain.from_storage(f"{src}/dogs/*", session=session)
    cats = DataChain.from_storage(f"{src}/cats/*", session=session)

    signal_default_value = Int.default_value(catalog.warehouse.db.dialect)

    dogs_and_cats = dogs | cats
    dogs1 = dogs.map(sig1=lambda: 1, output={"sig1": int})
    cats1 = cats.map(sig2=lambda: 2, output={"sig2": int})
    dogs2 = dogs_and_cats.merge(dogs1, "file.path", inner=inner1)
    cats2 = dogs_and_cats.merge(cats1, "file.path", inner=inner2)
    merged = dogs2.merge(cats2, "file.path", inner=inner3)

    merged_signals = (
        merged.select("file.path", "sig1", "sig2").order_by("file.path").results()
    )

    if inner1 and inner2 and inner3:
        assert merged_signals == []
    elif inner1:
        assert merged_signals == [
            ("dogs/dog1", 1, signal_default_value),
            ("dogs/dog2", 1, signal_default_value),
            ("dogs/dog3", 1, signal_default_value),
            ("dogs/others/dog4", 1, signal_default_value),
        ]
    elif inner2 and inner3:
        assert merged_signals == [
            ("cats/cat1", signal_default_value, 2),
            ("cats/cat2", signal_default_value, 2),
        ]
    else:
        assert merged_signals == [
            ("cats/cat1", signal_default_value, 2),
            ("cats/cat2", signal_default_value, 2),
            ("dogs/dog1", 1, signal_default_value),
            ("dogs/dog2", 1, signal_default_value),
            ("dogs/dog3", 1, signal_default_value),
            ("dogs/others/dog4", 1, signal_default_value),
        ]
