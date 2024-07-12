import datetime
import math
from collections.abc import Generator, Iterator

import numpy as np
import pandas as pd
import pytest

from datachain.lib.dc import C, DataChain
from datachain.lib.feature import Feature
from datachain.lib.file import File
from datachain.lib.signal_schema import (
    SignalResolvingError,
    SignalResolvingTypeError,
    SignalSchema,
)
from datachain.lib.udf_signature import UdfSignatureError
from datachain.lib.utils import DataChainParamsError

DF_DATA = {
    "first_name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "age": [25, 30, 35, 40, 45],
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
}

DF_OTHER_DATA = {
    "last_name": ["Smith", "Jones"],
    "country": ["USA", "Russia"],
}


class MyFr(Feature):
    nnn: str
    count: int


class MyNested(Feature):
    label: str
    fr: MyFr


features = [MyFr(nnn="n1", count=3), MyFr(nnn="n2", count=5), MyFr(nnn="n1", count=1)]
features_nested = [
    MyNested(fr=fr, label=f"label_{num}") for num, fr in enumerate(features)
]


def test_pandas_conversion(catalog):
    df = pd.DataFrame(DF_DATA)
    df1 = DataChain.from_pandas(df)
    df1 = df1.select("first_name", "age", "city").to_pandas()
    assert df1.equals(df)


def test_pandas_file_column_conflict(catalog):
    file_records = {"name": ["aa.txt", "bb.txt", "ccc.jpg", "dd", "e.txt"]}
    with pytest.raises(DataChainParamsError):
        DataChain.from_pandas(pd.DataFrame(DF_DATA | file_records))

    file_records = {"etag": [1, 2, 3, 4, 5]}
    with pytest.raises(DataChainParamsError):
        DataChain.from_pandas(pd.DataFrame(DF_DATA | file_records))


def test_pandas_uppercase_columns(catalog):
    data = {
        "FirstName": ["Alice", "Bob", "Charlie", "David", "Eva"],
        "Age": [25, 30, 35, 40, 45],
        "City": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
    }
    df = DataChain.from_pandas(pd.DataFrame(data)).to_pandas()
    assert all(col not in df.columns for col in data)
    assert all(col.lower() in df.columns for col in data)


def test_pandas_incorrect_column_names(catalog):
    with pytest.raises(DataChainParamsError):
        DataChain.from_pandas(
            pd.DataFrame({"First Name": ["Alice", "Bob", "Charlie", "David", "Eva"]})
        )

    with pytest.raises(DataChainParamsError):
        DataChain.from_pandas(
            pd.DataFrame({"": ["Alice", "Bob", "Charlie", "David", "Eva"]})
        )

    with pytest.raises(DataChainParamsError):
        DataChain.from_pandas(
            pd.DataFrame({"First@Name": ["Alice", "Bob", "Charlie", "David", "Eva"]})
        )


def test_from_features_basic(catalog):
    ds = DataChain.create_empty(DataChain.DEFAULT_FILE_RECORD)
    ds = ds.gen(lambda prm: [File(name="")] * 5, params="parent", output={"file": File})

    ds_name = "my_ds"
    ds.save(ds_name)
    ds = DataChain(name=ds_name)

    assert isinstance(ds.feature_schema, dict)
    assert isinstance(ds.signals_schema, SignalSchema)
    assert ds.schema.keys() == {"file"}
    assert set(ds.schema.values()) == {File}


def test_from_features(catalog):
    ds = DataChain.create_empty(DataChain.DEFAULT_FILE_RECORD)
    ds = ds.gen(
        lambda prm: list(zip([File(name="")] * len(features), features)),
        params="parent",
        output={"file": File, "t1": MyFr},
    )
    df1 = ds.to_pandas()

    assert df1[["t1.nnn", "t1.count"]].equals(
        pd.DataFrame({"t1.nnn": ["n1", "n2", "n1"], "t1.count": [3, 5, 1]})
    )


def test_dataset_registry(catalog):
    ds = DataChain.dataset_registry()
    datasets = [d for d in ds.iterate_one("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 0

    DataChain.from_features(fib=[1, 1, 2, 3, 5, 8]).save("fibonacci")

    ds = DataChain.dataset_registry()
    datasets = [d for d in ds.iterate_one("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6


def test_preserve_feature_schema(catalog):
    ds = DataChain.create_empty(DataChain.DEFAULT_FILE_RECORD)
    ds = ds.gen(
        lambda prm: list(zip([File(name="")] * len(features), features, features)),
        params="parent",
        output={"file": File, "t1": MyFr, "t2": MyFr},
    )

    ds_name = "my_ds1"
    ds.save(ds_name)
    ds = DataChain(name=ds_name)

    assert isinstance(ds.feature_schema, dict)
    assert isinstance(ds.signals_schema, SignalSchema)
    assert ds.schema.keys() == {"t1", "t2", "file"}
    assert set(ds.schema.values()) == {MyFr, File}


def test_from_features_simple_types(catalog):
    fib = [1, 1, 2, 3, 5, 8]
    values = ["odd" if num % 2 else "even" for num in fib]

    ds = DataChain.from_features(fib=fib, odds=values)

    df = ds.to_pandas()
    assert len(df) == len(fib)
    assert df["fib"].tolist() == fib
    assert df["odds"].tolist() == values


def test_from_features_more_simple_types(catalog):
    ds_name = "my_ds_type"
    DataChain.from_features(
        t1=features,
        num=range(len(features)),
        bb=[True, True, False],
        dd=[{}, {"ee": 3}, {"ww": 1, "qq": 2}],
        time=[
            datetime.datetime.now(),
            datetime.datetime.today(),
            datetime.datetime.today(),
        ],
        f=[3.14, 2.72, 1.62],
    ).save(ds_name)

    ds = DataChain(name=ds_name)
    assert ds.schema.keys() == {
        "t1",
        "num",
        "bb",
        "dd",
        "time",
        "f",
    }
    assert set(ds.schema.values()) == {
        MyFr,
        int,
        bool,
        dict,
        datetime.datetime,
        float,
    }


def test_file_list(catalog):
    names = ["f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json"]
    sizes = [1, 2, 3, 4, 5]
    files = [File(name=name, size=size) for name, size in zip(names, sizes)]

    ds = DataChain.from_features(file=files)

    for i, values in enumerate(ds.iterate()):
        assert values[0] == files[i]


def test_gen(catalog):
    class _TestFr(Feature):
        file: File
        sqrt: float
        my_name: str

    ds = DataChain.from_features(t1=features)
    ds = ds.gen(
        x=lambda m_fr: [
            _TestFr(
                file=File(name=""),
                sqrt=math.sqrt(m_fr.count),
                my_name=m_fr.nnn,
            )
        ],
        params="t1",
        output={"x": _TestFr},
    )

    df = ds.to_pandas()

    assert df["x.my_name"].tolist() == ["n1", "n2", "n1"]
    assert np.allclose(df["x.sqrt"], [math.sqrt(x) for x in [3, 5, 1]])
    with pytest.raises(KeyError):
        df["x.t1.nnn"]


def test_map(catalog):
    class _TestFr(Feature):
        sqrt: float
        my_name: str

    ds = DataChain.from_features(t1=features)

    df = ds.map(
        x=lambda m_fr: _TestFr(
            sqrt=math.sqrt(m_fr.count),
            my_name=m_fr.nnn + "_suf",
        ),
        params="t1",
        output={"x": _TestFr},
    ).to_pandas()

    assert df["x.my_name"].tolist() == ["n1_suf", "n2_suf", "n1_suf"]
    assert np.allclose(df["x.sqrt"], [math.sqrt(x) for x in [3, 5, 1]])


def test_agg(catalog):
    class _TestFr(Feature):
        f: File
        cnt: int
        my_name: str

    df = (
        DataChain.from_features(t1=features)
        .agg(
            x=lambda frs: [
                _TestFr(
                    f=File(name=""),
                    cnt=sum(f.count for f in frs),
                    my_name="-".join([fr.nnn for fr in frs]),
                )
            ],
            partition_by=C.t1.nnn,
            params="t1",
            output={"x": _TestFr},
        )
        .to_pandas()
    )

    assert len(df) == 2
    assert df["x.my_name"].tolist() == ["n1-n1", "n2"]
    assert df["x.cnt"].tolist() == [4, 5]


def test_agg_two_params(catalog):
    class _TestFr(Feature):
        f: File
        cnt: int
        my_name: str

    features2 = [
        MyFr(nnn="n1", count=6),
        MyFr(nnn="n2", count=10),
        MyFr(nnn="n1", count=2),
    ]

    ds = DataChain.from_features(t1=features, t2=features2).agg(
        x=lambda frs1, frs2: [
            _TestFr(
                f=File(name=""),
                cnt=sum(f1.count + f2.count for f1, f2 in zip(frs1, frs2)),
                my_name="-".join([fr.nnn for fr in frs1]),
            )
        ],
        partition_by=C.t1.nnn,
        params=("t1", "t2"),
        output={"x": _TestFr},
    )

    df = ds.to_pandas()
    assert len(df) == 2
    assert df["x.my_name"].tolist() == ["n1-n1", "n2"]
    assert df["x.cnt"].tolist() == [12, 15]


def test_agg_simple_iterator(catalog):
    def func(key, val) -> Iterator[tuple[File, str]]:
        for i in range(val):
            yield File(name=""), f"{key}_{i}"

    keys = ["a", "b", "c"]
    values = [3, 1, 2]
    ds = DataChain.from_features(key=keys, val=values).gen(res=func)

    df = ds.to_pandas()
    res = df["res_1"].tolist()
    assert res == ["a_0", "a_1", "a_2", "b_0", "c_0", "c_1"]


def test_agg_simple_iterator_error(catalog):
    chain = DataChain.from_features(key=["a", "b", "c"])

    with pytest.raises(UdfSignatureError):

        def func(key) -> int:
            return 1

        chain.gen(res=func)

    with pytest.raises(UdfSignatureError):

        class _MyCls(Feature):
            x: int

        def func(key) -> _MyCls:  # type: ignore[misc]
            return _MyCls(x=2)

        chain.gen(res=func)

    with pytest.raises(UdfSignatureError):

        def func(key) -> tuple[File, str]:  # type: ignore[misc]
            yield None, "qq"

        chain.gen(res=func)


def test_agg_tuple_result_iterator(catalog):
    class _ImageGroup(Feature):
        name: str
        size: int

    def func(key, val) -> Iterator[tuple[File, _ImageGroup]]:
        n = "-".join(key)
        v = sum(val)
        yield File(name=n), _ImageGroup(name=n, size=v)

    keys = ["n1", "n2", "n1"]
    values = [1, 5, 9]
    ds = DataChain.from_features(key=keys, val=values).agg(
        x=func, partition_by=C("key")
    )

    df = ds.to_pandas()
    assert len(df) == 2
    assert df["x_1.name"].tolist() == ["n1-n1", "n2"]
    assert df["x_1.size"].tolist() == [10, 5]


def test_agg_tuple_result_generator(catalog):
    class _ImageGroup(Feature):
        name: str
        size: int

    def func(key, val) -> Generator[tuple[File, _ImageGroup], None, None]:
        n = "-".join(key)
        v = sum(val)
        yield File(name=n), _ImageGroup(name=n, size=v)

    keys = ["n1", "n2", "n1"]
    values = [1, 5, 9]
    ds = DataChain.from_features(key=keys, val=values).agg(
        x=func, partition_by=C("key")
    )

    df = ds.to_pandas()
    assert len(df) == 2
    assert df["x_1.name"].tolist() == ["n1-n1", "n2"]
    assert df["x_1.size"].tolist() == [10, 5]


def test_iterate(catalog):
    dc = DataChain.from_features(f1=features, num=range(len(features)))

    n = 0
    for sample in dc.iterate():
        assert len(sample) == 2
        fr, num = sample

        assert isinstance(fr, MyFr)
        assert isinstance(num, int)
        assert num == n
        assert fr == features[n]

        n += 1

    assert n == len(features)


def test_iterate_nested_feature(catalog):
    dc = DataChain.from_features(sign1=features_nested)

    for n, sample in enumerate(dc.iterate()):
        assert len(sample) == 1
        nested = sample[0]

        assert isinstance(nested, MyNested)
        assert nested == features_nested[n]


def test_select_feature(catalog):
    dc = DataChain.from_features(my_n=features_nested)

    samples = dc.select("my_n").iterate()
    n = 0
    for sample in samples:
        assert sample[0] == features_nested[n]
        n += 1
    assert n == len(features_nested)

    samples = dc.select("my_n.fr").iterate()
    n = 0
    for sample in samples:
        assert sample[0] == features[n]
        n += 1
    assert n == len(features_nested)

    samples = dc.select("my_n.label", "my_n.fr.count").iterate()
    n = 0
    for sample in samples:
        label, count = sample
        assert label == features_nested[n].label
        assert count == features_nested[n].fr.count
        n += 1
    assert n == len(features_nested)


def test_select_columns_intersection(catalog):
    dc = DataChain.from_features(my_n=features_nested)

    samples = dc.select("my_n.fr", "my_n.fr.count").iterate()
    n = 0
    for sample in samples:
        fr, count = sample
        assert fr == features_nested[n].fr
        assert count == features_nested[n].fr.count
        n += 1
    assert n == len(features_nested)


def test_select_except(catalog):
    dc = DataChain.from_features(fr1=features_nested, fr2=features)

    samples = dc.select_except("fr2").iterate()
    n = 0
    for sample in samples:
        fr = sample[0]
        assert fr == features_nested[n]
        n += 1
    assert n == len(features_nested)


def test_select_wrong_type(catalog):
    dc = DataChain.from_features(fr1=features_nested, fr2=features)

    with pytest.raises(SignalResolvingTypeError):
        list(dc.select(4).iterate())

    with pytest.raises(SignalResolvingTypeError):
        list(dc.select_except(features[0]).iterate())


def test_select_except_error(catalog):
    dc = DataChain.from_features(fr1=features_nested, fr2=features)

    with pytest.raises(SignalResolvingError):
        list(dc.select_except("not_exist", "file").iterate())

    with pytest.raises(SignalResolvingError):
        list(dc.select_except("fr1.label", "file").iterate())


def test_select_restore_from_saving(catalog):
    dc = DataChain.from_features(my_n=features_nested)

    name = "test_test_select_save"
    dc.select("my_n.fr").save(name)

    restored = DataChain.from_dataset(name)
    n = 0
    restored_sorted = sorted(restored.iterate(), key=lambda x: x[0].count)
    features_sorted = sorted(features, key=lambda x: x.count)
    for sample in restored_sorted:
        assert sample[0] == features_sorted[n]
        n += 1
    assert n == len(features_nested)


def test_chain_of_maps(catalog):
    dc = (
        DataChain.from_features(my_n=features_nested)
        .map(full_name=lambda my_n: my_n.label + "-" + my_n.fr.nnn, output=str)
        .map(square=lambda my_n: my_n.fr.count**2, output=int)
    )

    signals = ["my_n", "full_name", "square"]
    assert len(dc.schema) == len(signals)
    for signal in signals:
        assert signal in dc.schema

    preserved = dc.save()
    for signal in signals:
        assert signal in preserved.schema


def test_vector(catalog):
    vector = [3.14, 2.72, 1.62]

    def get_vector(key) -> list[float]:
        return vector

    ds = DataChain.from_features(key=[123]).map(emd=get_vector)

    df = ds.to_pandas()
    assert np.allclose(df["emd"].tolist()[0], vector)


def test_vector_of_vectors(catalog):
    vector = [[3.14, 2.72, 1.62], [1.0, 2.0, 3.0]]

    def get_vector(key) -> list[list[float]]:
        return vector

    ds = DataChain.from_features(key=[123]).map(emd_list=get_vector)

    df = ds.to_pandas()
    actual = df["emd_list"].tolist()[0]
    assert len(actual) == 2
    assert np.allclose(actual[0], vector[0])
    assert np.allclose(actual[1], vector[1])


def test_unsupported_output_type(catalog):
    vector = [3.14, 2.72, 1.62]

    def get_vector(key) -> list[np.float64]:
        return [vector]

    with pytest.raises(TypeError):
        DataChain.from_features(key=[123]).map(emd=get_vector)


def test_collect_one(catalog):
    names = ["f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json"]
    sizes = [1, 2, 3, 4, 5]
    files = [File(name=name, size=size) for name, size in zip(names, sizes)]

    scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    chain = DataChain.from_features(file=files, score=scores)

    assert chain.collect_one("file") == files
    assert chain.collect_one("file.name") == names
    assert chain.collect_one("file.size") == sizes
    assert chain.collect_one("file.source") == [""] * len(names)
    assert np.allclose(chain.collect_one("score"), scores)

    for actual, expected in zip(
        chain.collect("file.size", "score"), [[x, y] for x, y in zip(sizes, scores)]
    ):
        assert len(actual) == 2
        assert actual[0] == expected[0]
        assert math.isclose(actual[1], expected[1], rel_tol=1e-7)


def test_default_output_type(catalog):
    names = ["f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json"]
    suffix = "-new"

    chain = DataChain.from_features(name=names).map(res1=lambda name: name + suffix)

    assert chain.collect_one("res1") == [t + suffix for t in names]


def test_create_model(catalog):
    chain = DataChain.from_features(name=["aaa", "b", "c"], count=[1, 4, 6])

    cls = chain.create_model("TestModel")
    assert isinstance(cls, type(Feature))

    fields = {n: f_info.annotation for n, f_info in cls.model_fields.items()}
    assert fields == {"name": str, "count": int}


def test_parse_tabular(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    dc = DataChain.from_storage(path.as_uri()).parse_tabular()
    df1 = dc.select("first_name", "age", "city").to_pandas()

    assert df1.equals(df)


def test_parse_tabular_format(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    dc = DataChain.from_storage(path.as_uri()).parse_tabular(format="json")
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert df1.equals(df)


def test_parse_tabular_empty(tmp_dir, catalog):
    path = tmp_dir / "test.parquet"
    with pytest.raises(DataChainParamsError):
        DataChain.from_storage(path.as_uri()).parse_tabular()


def test_parse_tabular_unify_schema(tmp_dir, catalog):
    df1 = pd.DataFrame(DF_DATA)
    df2 = pd.DataFrame(DF_OTHER_DATA)
    path1 = tmp_dir / "df1.parquet"
    path2 = tmp_dir / "df2.parquet"
    df1.to_parquet(path1)
    df2.to_parquet(path2)
    df_combined = (
        pd.concat([df1, df2], ignore_index=True)
        .replace({"": None, 0: None, np.nan: None})
        .sort_values("first_name")
        .reset_index(drop=True)
    )
    dc = (
        DataChain.from_storage(tmp_dir.as_uri())
        .filter(C("name").glob("*.parquet"))
        .parse_tabular()
    )
    df = dc.select("first_name", "age", "city", "last_name", "country").to_pandas()
    df = (
        df.replace({"": None, 0: None, np.nan: None})
        .sort_values("first_name")
        .reset_index(drop=True)
    )
    assert df.equals(df_combined)


def test_parse_tabular_output(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    output = {"fname": str, "age": int, "loc": str}
    dc = DataChain.from_storage(path.as_uri()).parse_tabular(
        format="json", output=output
    )
    df1 = dc.select("fname", "age", "loc").to_pandas()
    df.columns = ["fname", "age", "loc"]
    assert df1.equals(df)


def test_parse_csv(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path)
    dc = DataChain.from_storage(path.as_uri()).parse_csv()
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert df1.equals(df)


def test_parse_csv_no_header_error(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    with pytest.raises(DataChainParamsError):
        DataChain.from_storage(path.as_uri()).parse_csv(header=False)


def test_parse_csv_no_header_output(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    dc = DataChain.from_storage(path.as_uri()).parse_csv(
        header=False, output={"first_name": str, "age": int, "city": str}
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert (df1.values != df.values).sum() == 0


def test_parse_csv_no_header_column_names(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    dc = DataChain.from_storage(path.as_uri()).parse_csv(
        header=False, column_names=["first_name", "age", "city"]
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert (df1.values != df.values).sum() == 0


def test_parse_csv_column_names_and_output(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path)
    column_names = ["fname", "age", "loc"]
    output = {"fname": str, "age": int, "loc": str}
    with pytest.raises(DataChainParamsError):
        DataChain.from_storage(path.as_uri()).parse_csv(
            column_names=column_names, output=output
        )


def test_parse_csv_tab_delimited(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, sep="\t")
    dc = DataChain.from_storage(path.as_uri()).parse_csv(delimiter="\t")
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert df1.equals(df)


def test_parse_parquet_partitioned(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path, partition_cols=["first_name"])
    dc = DataChain.from_storage(path.as_uri()).parse_parquet()
    df1 = dc.select("first_name", "age", "city").to_pandas()
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    assert df1.equals(df)


def test_parse_parquet_filter_partitions(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path, partition_cols=["first_name"])
    dc = (
        DataChain.from_storage(path.as_uri())
        .filter(C("parent").glob("*first_name=Alice*"))
        .parse_parquet()
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    assert df1.equals(df.loc[:0])


@pytest.mark.parametrize("processes", [False, 2, True])
def test_parallel(processes, catalog):
    prefix = "t & "
    vals = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    res = (
        DataChain.from_features(key=vals)
        .settings(parallel=processes)
        .map(res=lambda key: prefix + key)
        .collect_one("res")
    )

    assert res == [prefix + v for v in vals]


def test_exec(catalog):
    names = ("f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json")
    all_names = set()

    dc = (
        DataChain.from_features(name=names)
        .map(nop=lambda name: all_names.add(name))
        .exec()
    )
    assert isinstance(dc, DataChain)
    assert all_names == set(names)


def test_extend_features(catalog):
    dc = DataChain.from_features(f1=features, num=range(len(features)))

    res = dc._extend_features("select", "num")
    assert isinstance(res, DataChain)
    assert res.signals_schema.values == {"num": int}

    res = dc._extend_features("sum", "num")
    assert res == sum(range(len(features)))
