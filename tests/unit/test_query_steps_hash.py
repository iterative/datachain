import pytest
import sqlalchemy as sa

from datachain import C, func
from datachain.func.func import Func
from datachain.lib.signal_schema import SignalSchema
from datachain.query.dataset import (
    SQLFilter,
    SQLMutate,
    SQLSelect,
    SQLSelectExcept,
)


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
