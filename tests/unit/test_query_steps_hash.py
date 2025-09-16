import math
from dataclasses import replace

import pytest
import sqlalchemy as sa
from pydantic import BaseModel

import datachain as dc
from datachain import C, func
from datachain.func.func import Func
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf import Aggregator, Generator, Mapper
from datachain.lib.udf_signature import UdfSignature
from datachain.query.dataset import (
    QueryStep,
    RowGenerator,
    SQLCount,
    SQLDistinct,
    SQLFilter,
    SQLGroupBy,
    SQLJoin,
    SQLLimit,
    SQLMutate,
    SQLOffset,
    SQLOrderBy,
    SQLSelect,
    SQLSelectExcept,
    SQLUnion,
    Subtract,
    UDFSignal,
)


class CustomFeature(BaseModel):
    sqrt: float
    my_name: str


def double(x):
    return x * 2


def double_gen(x):
    yield x * 2


def double_gen_multi_arg(x, y):
    yield x * 2
    yield y * 2


def custom_feature_gen(m_fr):
    yield CustomFeature(
        sqrt=math.sqrt(m_fr.count),
        my_name=m_fr.nnn + "_suf",
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
    "inputs,_hash",
    [
        (
            (C("name"), C("age") * 10, func.avg("id"), C("country").label("country")),
            "d03395827dcdddc2b2c3f0a3dafb71affa89c7f3b03b89e42734af2aea0e05ba",
        ),
        ((), "3245ba76bc1e4b1b1d4d775b88448ff02df9473bd919929166c70e9e2b245345"),
        (
            (C("name"),),
            "fe30656afd177ef32da191cc5ab3c68268282c382ef405d753e128b69767602f",
        ),
        (("name",), "46eeec88c5f842bd478d3ec87032c49b22adcdd46572463b0acde4b2bac0900a"),
    ],
)
def test_select_hash(inputs, _hash):
    assert SQLSelect(inputs).hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
    [
        (
            (C("name"), C("age") * 10, func.avg("id"), C("country").label("country")),
            "19894de08d545f3db85242be292dea0bb1ef47b0feaaf2c9359b159c7aa588c6",
        ),
        ((), "0d27e4cfa3801628afc535190c64a426d9db66e5145c57129b9f5ca0935ef29e"),
        (
            (C("name"),),
            "9515589e525bfa21cec0b68edf41c09e8df26e5c3023fd0775ba0ea02c9f6c8f",
        ),
        (("name",), "e26923a0433e549e680a4bcbc5cb95bb9a523c4b47ae23b07b2a928a609fc498"),
    ],
)
def test_select_except_hash(inputs, _hash):
    assert SQLSelectExcept(inputs).hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
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
def test_filter_hash(inputs, _hash):
    assert SQLFilter(inputs).hash() == _hash


@pytest.mark.parametrize(
    "inputs,schema,_hash",
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
def test_mutate_hash(inputs, schema, _hash):
    # transforming input into format SQLMutate expects
    inputs = (
        v.label(k).get_column(schema) if isinstance(v, Func) else v.label(k)
        for k, v in inputs.items()
    )
    assert SQLMutate(inputs, new_schema=None).hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
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
def test_order_by_hash(inputs, _hash):
    assert SQLOrderBy(inputs).hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
    [
        (5, "9fc462c7b5fe66106c8056b9f361817523de5c9f8d4e4b847e79cb02feba1351"),
        (0, "1da7ad424bfdb853e852352fbb853722eb5fdc119592a778679aa00ba29f971a"),
    ],
)
def test_limit_hash(inputs, _hash):
    assert SQLLimit(inputs).hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
    [
        (5, "ff65be6bef149f6f2568f33c2bd0ac3362018a504caadf52c221a2e64acc5bb3"),
        (0, "e88121711a1fa5da46ea2305e0d6fbeebe63f5b575450c628e7bf6f81e73aa46"),
    ],
)
def test_offset_hash(inputs, _hash):
    assert SQLOffset(inputs).hash() == _hash


@pytest.mark.parametrize(
    "_hash",
    [
        "8867973da58bd4d14c023fa9bad98dc50c18ba69240347216f7a8a1c7e70d377",
        "8867973da58bd4d14c023fa9bad98dc50c18ba69240347216f7a8a1c7e70d377",
    ],
)
def test_count_hash(_hash):
    assert SQLCount().hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
    [
        (("name",), "bb0a1acba3bce39d31cc05dc01e57fc7265e451154187a6f93fbcf2001525c51"),
        (
            ("name", "age"),
            "29203756f44599f2728c70d75d92ff7af6110c8602e25839127c736d25a30c4b",
        ),
        ((), "7d4efeefbe9d1694bb89e7bf8b2d3f1d96ed0603e312b48d247d0ed3c881bf48"),
    ],
)
def test_distinct_hash(inputs, _hash):
    assert SQLDistinct(inputs, dialect=None).hash() == _hash


def test_union_hash(test_session, numbers_dataset):
    chain1 = dc.read_dataset("numbers").filter(C("num") > 50).limit(10)
    chain2 = dc.read_dataset("numbers").filter(C("num") < 50).limit(20)

    assert SQLUnion(chain1._query, chain2._query).hash() == (
        "ac9c210ba6c599d5ce8155692fccb5d56ec45562d87aedbc7853a283739880b3"
    )


@pytest.mark.parametrize(
    "predicates,inner,full,rname,_hash",
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
    test_session, numbers_dataset, predicates, inner, full, rname, _hash
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
        == _hash
    )


@pytest.mark.parametrize(
    "columns,partition_by,_hash",
    [
        (
            {"cnt": func.count(), "sum": func.sum("id")},
            [
                C("id"),
            ],
            "0f28ac6aa6daee1892d5e79b559c9c1c2072cec2d53d4e0f12c3ae42db1a869f",
        ),
        (
            {"cnt": func.count(), "sum": func.sum("id")},
            [C("id"), C("name")],
            "f8ef71fc6d3438cd6905e0a4d96f9b13a465c4a955127d929837e3f0ac3d31d6",
        ),
        (
            {"cnt": func.count()},
            [],
            "fe833a3ce997c919bcf3a2c5de1e76f2481a0937320f9fa0c2a8b3c191cea480",
        ),
    ],
)
def test_group_by_hash(columns, partition_by, _hash):
    schema = SignalSchema({"id": int})
    # transforming inputs into format SQLGroupBy expects
    columns = [v.get_column(schema, label=k) for k, v in columns.items()]
    assert SQLGroupBy(columns, partition_by).hash() == _hash


@pytest.mark.parametrize(
    "on,_hash",
    [
        (
            [("id", "id")],
            "416c1de08bcb423fbcf300d65b86387d83133d5d8e8d285fb09fec9b748165e8",
        ),
        (
            [("id", "id"), ("name", "name")],
            "f30681465b8a6e9c7f3c752fc0c12e7a642ca78077cd36b66075209665e9f0ac",
        ),
        (
            [],
            "c53881a69a3bb95de8d7348b895a443978b48fa3e7f5129d40719d1a74db16e2",
        ),
    ],
)
def test_subtract_hash(test_session, numbers_dataset, on, _hash):
    chain = dc.read_dataset("numbers").filter(C("num") > 50).limit(20)
    assert Subtract(chain._query, test_session.catalog, on).hash() == _hash


@pytest.mark.parametrize(
    "func,params,output,_hash",
    [
        (
            lambda x: x * 2,
            ["x"],
            {"double": int},
            "decb3165f496aa7df8203d27d58b884a60b2dcec25edcc02d251be78c5cdf834",
        ),
        (
            lambda y: y * 2,
            ["y"],
            {"double": int},
            "47d64007cf9cd8d46efe75e9f18b75a8f974634a5eea739d95692dd499ee247e",
        ),
        (
            double,
            ["x"],
            {"double": int},
            "5b38846a0582d96d20d6eb7b633c1f6024aabd98dab86bb0636d33e12b4232cd",
        ),
        (
            lambda m_fr: CustomFeature(
                sqrt=math.sqrt(m_fr.count),
                my_name=m_fr.nnn + "_suf",
            ),
            ["t1"],
            {"x": CustomFeature},
            "c877e5f489a3adbe9fa94df79767d24dc6b64b404f68d910ea59fe4a8510cb60",
        ),
    ],
)
def test_udf_mapper_hash(
    func,
    params,
    output,
    _hash,
):
    sign = UdfSignature.parse("", {}, func, params, output, False)
    udf_adapter = Mapper._create(sign, SignalSchema(sign.params)).to_udf_wrapper()
    assert UDFSignal(udf_adapter, None).hash() == _hash


@pytest.mark.parametrize(
    "func,params,output,_hash",
    [
        (
            double_gen,
            ["x"],
            {"double": int},
            "ecf2a08ba539c174857215b563933612830fb311eb3c030d7bc1deb1bbe24c55",
        ),
        (
            double_gen_multi_arg,
            ["x", "y"],
            {"double": int},
            "e6ab21fef879b0e5992f27c2834e34fedccd8ffed9b668593c5cf4efd3431929",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            "967d649d5385c8d3d1231daa57d066e0c1dfcbb02c871a57960ed148e5223317",
        ),
    ],
)
def test_udf_generator_hash(
    func,
    params,
    output,
    _hash,
):
    sign = UdfSignature.parse("", {}, func, params, output, False)
    udf_adapter = Generator._create(sign, SignalSchema(sign.params)).to_udf_wrapper()
    assert RowGenerator(udf_adapter, None).hash() == _hash


@pytest.mark.parametrize(
    "func,params,output,partition_by,_hash",
    [
        (
            double_gen,
            ["x"],
            {"double": int},
            [C("x")],
            "9ade6a992886adad25d3ab1e00cea23f3dc5d15cf27a86558fd744646393bad1",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            [C.t1.my_name],
            "73968e35606d694e6cd4b98c168560427e888f66fa9c1b4f90ed6af8ec482795",
        ),
    ],
)
def test_udf_aggregator_hash(
    func,
    params,
    output,
    partition_by,
    _hash,
):
    sign = UdfSignature.parse("", {}, func, params, output, False)
    udf_adapter = Aggregator._create(sign, SignalSchema(sign.params)).to_udf_wrapper()
    assert RowGenerator(udf_adapter, None, partition_by=partition_by).hash() == _hash


@pytest.mark.parametrize(
    "namespace_name,project_name,name,version,_hash",
    [
        (
            "default",
            "default",
            "numbers",
            "1.0.4",
            "8173fb1d88df5cca3e904cbd17a9b80a0c8a682425c32cd95e32e1e196b7eff8",
        ),
        (
            "dev",
            "animals",
            "cats",
            "1.0.1",
            "e0aec7fe323ae3482ee2e74030a87ebb73dbb823ce970e15fdfcbd43e7abe2da",
        ),
        (
            "system",
            "listing",
            "lst__s3://bucket",
            "1.0.1",
            "19dff9f21030312c7469de7284cac2841063c22c62a7948a68f25ca018777c6d",
        ),
    ],
)
def test_query_step_hash(
    dataset_record, namespace_name, project_name, name, version, _hash
):
    namespace = replace(dataset_record.project.namespace, name=namespace_name)
    project = dataset_record.project
    project = replace(project, namespace=namespace)
    project = replace(project, name=project_name)
    dataset_record.project = project
    dataset_record.name = name
    dataset_record.versions[0].version = version

    assert QueryStep(None, dataset_record, version).hash() == _hash
