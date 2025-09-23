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
        mock.return_value = ("lst__s3://my-bucket", "", "", True)
        yield mock


def test_read_values():
    pytest.skip(
        "Hash of the chain started with read_values is currently inconsistent,"
        " meaning it produces different hash every time. This happens because we"
        " create random name dataset in the process. Correct solution would be"
        " to calculate hash of all those input values."
    )
    assert dc.read_values(num=[1, 2, 3]).hash() == ""


def test_read_storage(mock_get_listing):
    assert dc.read_storage("s3://bucket").hash() == (
        "c38b6f4ebd7f0160d9f900016aad1e6781acd463f042588cfe793e9d189a8a0e"
    )


def test_read_dataset(test_session):
    dc.read_values(num=[1, 2, 3], session=test_session).save("dev.animals.cats")
    assert dc.read_dataset(
        name="dev.animals.cats", version="1.0.0", session=test_session
    ).hash() == ("51f2e5b81e40a22062a75c1590d0ccab880d182df9b39f610c6ccc503a5eb33c")


def test_order_of_steps(mock_get_listing):
    assert (
        dc.read_storage("s3://bucket").mutate(new=10).filter(C("age") > 20).hash()
    ) == "08a6c5657feaea55c734bc8e2b3eb0733ea692d4eab5fa78fa26409e6c2af098"

    assert (
        dc.read_storage("s3://bucket").filter(C("age") > 20).mutate(new=10).hash()
    ) == "e91b84094233a2bf4d08d6a95e55529a65d900399be3a05dc3e2ca0401f8f25b"


def test_all_possible_steps(test_session):
    persons_ds_name = "dev.my_pr.persons"
    players_ds_name = "dev.my_pr.players"

    def map_worker(person: Person) -> Worker:
        return Worker(
            name=person.name,
            age=person.age,
            title="worker",
        )

    def gen_persons(person):
        yield Person(
            age=person.age * 2,
            name=person.name + "_suf",
        )

    def agg_persons(persons):
        return PersonAgg(ages=sum(p.age for p in persons), name=persons[0].age)

    dc.read_values(person=persons, session=test_session).save(persons_ds_name)
    dc.read_values(player=players, session=test_session).save(players_ds_name)

    players_chain = dc.read_dataset(
        players_ds_name, version="1.0.0", session=test_session
    )

    assert (
        dc.read_dataset(persons_ds_name, version="1.0.0", session=test_session)
        .mutate(age_double=C("person.age") * 2)
        .filter(C("person.age") > 20)
        .order_by("person.name", "person.age")
        .gen(
            person=gen_persons,
            output=Person,
        )
        .map(
            worker=map_worker,
            params="person",
            output={"worker": Worker},
        )
        .agg(
            persons=agg_persons,
            partition_by=C.person.name,
            params="person",
            output={"persons": PersonAgg},
        )
        .merge(players_chain, "persons.name", "player.name")
        .distinct("persons.name")
        .sample(10)
        .offset(2)
        .limit(5)
        .group_by(age_avg=func.avg("persons.age"), partition_by="persons.name")
        .select("persons.name", "age_avg")
        .subtract(
            players_chain,
            on=["persons.name"],
            right_on=["player.name"],
        )
        .hash()
    ) == "44b231652aee9712444ee26d5ecc77e6b87f768d17e6b8333303764d3706413b"


def test_diff(test_session):
    persons_ds_name = "dev.my_pr.persons"
    players_ds_name = "dev.my_pr.players"

    dc.read_values(person=persons, session=test_session).save(persons_ds_name)
    dc.read_values(player=players, session=test_session).save(players_ds_name)

    players_chain = dc.read_dataset(
        players_ds_name, version="1.0.0", session=test_session
    )

    assert (
        dc.read_dataset(persons_ds_name, version="1.0.0", session=test_session)
        .diff(
            players_chain,
            on=["person.name"],
            right_on=["player.name"],
            status_col="diff",
        )
        .hash()
    ) == "aef929f3bf247966703534aa3daffb76fa8802d64660293deb95155ffacd8b77"
