import math
from typing import Optional

import pytest
from pydantic import BaseModel

from datachain.lib.dc import DataChain, DatasetMergeError
from datachain.lib.signal_schema import SignalResolvingError
from datachain.sql.types import Float, String


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


def test_merge_objects(catalog):
    ch1 = DataChain.from_values(emp=employees)
    ch2 = DataChain.from_values(team=team)
    ch = ch1.merge(ch2, "emp.person.name", "team.player")

    str_default = String.default_value(catalog.warehouse.db.dialect)
    float_default = Float.default_value(catalog.warehouse.db.dialect)

    i = 0
    j = 0
    for items in ch.collect():
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
            assert player.weight == float_default
            assert player.height == float_default

    assert i == len(employees)
    assert j == len(team)


def test_merge_similar_objects(catalog):
    new_employees = [
        Employee(id=152, person=User(name="Bob", age=27)),
        Employee(id=201, person=User(name="Karl", age=18)),
        Employee(id=154, person=User(name="David", age=29)),
    ]

    ch1 = DataChain.from_values(emp=employees)
    ch2 = DataChain.from_values(emp=new_employees)

    rname = "qq"
    ch = ch1.merge(ch2, "emp.person.name", rname=rname)

    assert list(ch.signals_schema.values.keys()) == ["sys", "emp", rname + "emp"]

    empl = list(ch.collect())
    assert len(empl) == 4
    assert len(empl[0]) == 2

    ch_inner = ch1.merge(ch2, "emp.person.name", rname=rname, inner=True)
    assert len(list(ch_inner.collect())) == 2


def test_merge_values(catalog):
    order_ids = [11, 22, 33, 44]
    order_descr = ["water", "water", "paper", "water"]

    delivery_ids = [11, 44]
    delivery_time = [24.0, 16.5]

    float_default = Float.default_value(catalog.warehouse.db.dialect)

    ch1 = DataChain.from_values(id=order_ids, descr=order_descr)
    ch2 = DataChain.from_values(id=delivery_ids, time=delivery_time)

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

        if time != float_default:
            assert id == delivery_ids[j]
            assert time == delivery_time[j]
            j += 1

    assert i == len(order_ids)
    assert j == len(delivery_ids)


def test_merge_multi_conditions(catalog):
    order_ids = [11, 22, 33, 44]
    order_name = ["water", "water", "paper", "water"]
    order_descr = ["still water", "still water", "white paper", "sparkling water"]

    delivery_ids = [11, 44]
    delivery_name = ["water", "unknown"]
    delivery_time = [24.0, 16.5]

    ch1 = DataChain.from_values(id=order_ids, name=order_name, descr=order_descr)
    ch2 = DataChain.from_values(
        id=delivery_ids, d_name=delivery_name, time=delivery_time
    )

    ch = ch1.merge(ch2, ("id", "name"), ("id", "d_name"))

    res = list(ch.collect())

    assert len(res) == max(len(employees), len(team))
    success_ids = set()
    for items in res:
        if items[3]:
            success_ids.add(items[0])

    assert success_ids == {11}


def test_merge_errors(catalog):
    ch1 = DataChain.from_values(emp=employees)
    ch2 = DataChain.from_values(team=team)

    with pytest.raises(SignalResolvingError):
        ch1.merge(ch2, "unknown")

    ch1.merge(ch2, ["emp.person.name"], ["team.sport"])
    ch1.merge(ch2, ["emp.person.name"], ["team.sport"])
    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, ["emp.person.name"], ["team.player", "team.sport"])

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, 33)

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, "emp.person.name", True)


def test_merge_with_itself(catalog):
    ch = DataChain.from_values(emp=employees)
    merged = ch.merge(ch, "emp.id")

    count = 0
    for left, right in merged.collect():
        assert isinstance(left, Employee)
        assert isinstance(right, Employee)
        assert left == right == employees[count]
        count += 1

    assert count == len(employees)
