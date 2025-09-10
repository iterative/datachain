import pytest
import sqlalchemy as sa

import datachain as dc
from datachain import C, func
from datachain.func.func import Func
from datachain.lib.signal_schema import SignalSchema
from datachain.query.dataset import (
    SQLCount,
    SQLDistinct,
    SQLFilter,
    SQLJoin,
    SQLLimit,
    SQLMutate,
    SQLOffset,
    SQLOrderBy,
    SQLSelect,
    SQLSelectExcept,
    SQLUnion,
)


@pytest.fixture
def numbers_dataset(test_session):
    """
    Fixture to create dataset with stable / constant UUID to have consistent
    hash values in tests as it goes into chain hash calculation
    """
    dc.read_values(num=list(range(100)), session=test_session).save("numbers")
    test_session.catalog.metastore.update_dataset_version(
        test_session.catalog.get_dataset("numbers"),
        "1.0.0",
        uuid="9045d46d-7c57-4442-aae3-3ca9e9f286c4",
    )

    return test_session.catalog.get_dataset("numbers")


@pytest.mark.parametrize(
    "inputs,result",
    [
        (
            (C("name"), C("age") * 10, func.avg("id"), C("country").label("country")),
            "f2f02f8197037661348d11a4bdf19bd7913d067d08ee1ff4eea08652f1a109d9",
        ),
        ((), "3245ba76bc1e4b1b1d4d775b88448ff02df9473bd919929166c70e9e2b245345"),
        (
            (C("name"),),
            "fe30656afd177ef32da191cc5ab3c68268282c382ef405d753e128b69767602f",
        ),
        (("name",), "46eeec88c5f842bd478d3ec87032c49b22adcdd46572463b0acde4b2bac0900a"),
    ],
)
def test_select_hash(inputs, result):
    assert SQLSelect(inputs).hash() == result


@pytest.mark.parametrize(
    "inputs,result",
    [
        (
            (C("name"), C("age") * 10, func.avg("id"), C("country").label("country")),
            "40af3a740d12d33e350c41b7c8d9651a2d9c560720d0e8474906650504cf3066",
        ),
        ((), "0d27e4cfa3801628afc535190c64a426d9db66e5145c57129b9f5ca0935ef29e"),
        (
            (C("name"),),
            "9515589e525bfa21cec0b68edf41c09e8df26e5c3023fd0775ba0ea02c9f6c8f",
        ),
        (("name",), "e26923a0433e549e680a4bcbc5cb95bb9a523c4b47ae23b07b2a928a609fc498"),
    ],
)
def test_select_except_hash(inputs, result):
    assert SQLSelectExcept(inputs).hash() == result


@pytest.mark.parametrize(
    "inputs,result",
    [
        (
            (sa.and_(C("name") != "John", C("age") * 10 > 100)),
            "ba98f1a292cc7e95402899a43e5392708bcf448332e060becb24956fb531bfd0",
        ),
        ((), "19e718af35ddc311aa892756fa4f95413ce17db7c8b27f68200d9c3ce0fc8dbf"),
        (
            (C("files.path").glob("*.jpg"),),
            "c77898b24747f5106fd3793862d6c227e0423e096c6859ac95c27a9f7f7a824b",
        ),
    ],
)
def test_filter_hash(inputs, result):
    assert SQLFilter(inputs).hash() == result


@pytest.mark.parametrize(
    "inputs,schema,result",
    [
        (
            {"new_id": func.sum("id")},
            SignalSchema({"id": int}),
            "d8e3af2fa2b5357643f80702455f0bbecb795b38bbb37eef24c644315e28617c",
        ),
        (
            {"new_id": C("id") * 10, "old_id": C("id")},
            SignalSchema({"id": int}),
            "beea21224d3e2fae077a6a38d663fbaea0549fd38508b48fac3454cd76eca0df",
        ),
        (
            {},
            SignalSchema({"id": int}),
            "b9717325e70a10ccd55c7faa22d5099ac8d5726d1a3c0eb3cfb001c7f628ce7f",
        ),
    ],
)
def test_mutate_hash(inputs, schema, result):
    # transforming input into format SQLMutate expects
    inputs = (
        v.label(k).get_column(schema) if isinstance(v, Func) else v.label(k)
        for k, v in inputs.items()
    )
    assert SQLMutate(inputs, new_schema=None).hash() == result


@pytest.mark.parametrize(
    "inputs,result",
    [
        (
            (C("name"), C("age")),
            "8368b3239fd66422c18d561d2b61dbbae9fd88f9c935f67719b0d12ada50ffb6",
        ),
        (("name",), "b3562b4508052e5a57bc84ae862255939df294eb079e124c5af61fc21044343e"),
        (
            (sa.desc(C("name")),),
            "fd91c8cfe480debf1cdcf2b3f91462393a75042d0752a813ecc65dfed1ac7a6c",
        ),
        ((), "c525013178ef24a807af6d4dd44d108c20a5224eb3ab88b84c55c635ec32ba04"),
    ],
)
def test_order_by_hash(inputs, result):
    assert SQLOrderBy(inputs).hash() == result


@pytest.mark.parametrize(
    "inputs,result",
    [
        (5, "9fc462c7b5fe66106c8056b9f361817523de5c9f8d4e4b847e79cb02feba1351"),
        (0, "1da7ad424bfdb853e852352fbb853722eb5fdc119592a778679aa00ba29f971a"),
    ],
)
def test_limit_hash(inputs, result):
    assert SQLLimit(inputs).hash() == result


@pytest.mark.parametrize(
    "inputs,result",
    [
        (5, "ff65be6bef149f6f2568f33c2bd0ac3362018a504caadf52c221a2e64acc5bb3"),
        (0, "e88121711a1fa5da46ea2305e0d6fbeebe63f5b575450c628e7bf6f81e73aa46"),
    ],
)
def test_offset_hash(inputs, result):
    assert SQLOffset(inputs).hash() == result


@pytest.mark.parametrize(
    "result",
    [
        "8867973da58bd4d14c023fa9bad98dc50c18ba69240347216f7a8a1c7e70d377",
        "8867973da58bd4d14c023fa9bad98dc50c18ba69240347216f7a8a1c7e70d377",
    ],
)
def test_count_hash(result):
    assert SQLCount().hash() == result


@pytest.mark.parametrize(
    "inputs,result",
    [
        (("name",), "bb0a1acba3bce39d31cc05dc01e57fc7265e451154187a6f93fbcf2001525c51"),
        (
            ("name", "age"),
            "29203756f44599f2728c70d75d92ff7af6110c8602e25839127c736d25a30c4b",
        ),
        ((), "7d4efeefbe9d1694bb89e7bf8b2d3f1d96ed0603e312b48d247d0ed3c881bf48"),
    ],
)
def test_distinct_hash(inputs, result):
    assert SQLDistinct(inputs, dialect=None).hash() == result


def test_union_hash(test_session, numbers_dataset):
    chain1 = dc.read_dataset("numbers").filter(C("num") > 50).limit(10)
    chain2 = dc.read_dataset("numbers").filter(C("num") < 50).limit(20)

    assert SQLUnion(chain1._query, chain2._query).hash() == (
        "ac9c210ba6c599d5ce8155692fccb5d56ec45562d87aedbc7853a283739880b3"
    )


@pytest.mark.parametrize(
    "predicates,inner,full,rname,result",
    [
        (
            "id",
            True,
            False,
            "{name}_right",
            "bcac0223d91650c419cea1626502dbb77bf3deabf6b522b0d0dc1cdafd8d488a",
        ),
        (
            ("id", "name"),
            False,
            True,
            "{name}_r",
            "035a972f3f830ceea1ef14781e2a2bdbaa3bf3db2320a5147826d1768bd64315",
        ),
    ],
)
def test_join_hash(
    test_session, numbers_dataset, predicates, inner, full, rname, result
):
    chain1 = dc.read_dataset("numbers").filter(C("num") > 50).limit(10)
    chain2 = dc.read_dataset("numbers").filter(C("num") < 50).limit(20)

    assert (
        SQLJoin(
            test_session.catalog,
            chain1._query,
            chain2._query,
            predicates,
            inner,
            full,
            rname,
        ).hash()
        == result
    )
