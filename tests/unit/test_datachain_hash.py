from unittest.mock import patch

import pytest
from pydantic import BaseModel

import datachain as dc
from datachain import func
from datachain.lib.dc import C


class Person(BaseModel):
    name: str
    age: float


class PersonAgg(BaseModel):
    name: str
    ages: float


class Player(BaseModel):
    name: str
    sport: str


class Worker(BaseModel):
    name: str
    age: float
    title: str


persons = [
    Person(name="p1", age=10),
    Person(name="p2", age=20),
    Person(name="p3", age=30),
    Person(name="p4", age=40),
    Person(name="p5", age=40),
    Person(name="p6", age=60),
]


players = [
    Player(name="p1", sport="baksetball"),
    Player(name="p2", sport="soccer"),
    Player(name="p3", sport="baseball"),
    Player(name="p4", sport="tennis"),
]


@pytest.fixture
def mock_get_listing():
    with patch("datachain.lib.dc.storage.get_listing") as mock:
        # Optionally set a default return value so you don't have to repeat it in tests
        mock.return_value = ("lst__s3://my-bucket", "", "", True)
        yield mock


def test_read_values():
    # hash from read values is currently inconsietent (evety time we get different
    # value).
    assert dc.read_values(num=[1, 2, 3]).hash()


def test_read_storage(mock_get_listing):
    assert dc.read_storage("s3://bucket").hash() == (
        "c38b6f4ebd7f0160d9f900016aad1e6781acd463f042588cfe793e9d189a8a0e"
    )


def test_read_dataset(test_session):
    dc.read_values(num=[1, 2, 3], session=test_session).save("cats")
    assert dc.read_dataset(
        name="cats", version="1.0.0", session=test_session
    ).hash() == ("54634c934f1d0d03bdd9409d0dcff3a6261921a78a0ebce4752bf96a16b99604")


def test_order_of_steps(mock_get_listing):
    assert (
        dc.read_storage("s3://bucket").mutate(new=10).filter(C("age") > 20).hash()
    ) == "08a6c5657feaea55c734bc8e2b3eb0733ea692d4eab5fa78fa26409e6c2af098"

    assert (
        dc.read_storage("s3://bucket").filter(C("age") > 20).mutate(new=10).hash()
    ) == "e91b84094233a2bf4d08d6a95e55529a65d900399be3a05dc3e2ca0401f8f25b"


def test_all_possible_steps(test_session):
    def gen_persons(person):
        yield Person(
            age=person.age * 2,
            name=person.name + "_suf",
        )

    dc.read_values(person=persons, session=test_session).save("persons")
    dc.read_values(player=players, session=test_session).save("players")

    players_chain = dc.read_dataset("players", version="1.0.0", session=test_session)

    assert (
        dc.read_dataset("persons", version="1.0.0", session=test_session)
        .mutate(age_double=C("person.age") * 2)
        .filter(C("person.age") > 20)
        .order_by("person.name", "person.age")
        .gen(
            person=gen_persons,
            output=Person,
        )
        .map(
            worker=lambda person: Worker(
                name=person.name,
                age=person.age,
                title="worker",
            ),
            params="person",
            output={"worker": Worker},
        )
        .agg(
            persons=lambda persons: [
                PersonAgg(ages=sum(p.age for p in persons), name=persons[0].age)
            ],
            partition_by=C.person.name,
            params="person",
            output={"persons": PersonAgg},
        )
        .merge(players_chain, "persons.name", "player.name")
        .distinct("persons.name")
        .sample(10)
        .offset(2)
        .limit(5)
        .group_by(avg=func.avg("persons.age"), partition_by="persons.name")
        .select("persons.name")
        .hash()
    ) == "db1b7bf05f771bddc85b67c69eece3098f89f60b7baad47d52b8fab8476fbfdb"
