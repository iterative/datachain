import datetime
import math
from collections.abc import Generator, Iterator
from unittest.mock import ANY

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from datachain import Column
from datachain.lib.dc import C, DataChain, Sys
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


class MyFr(BaseModel):
    nnn: str
    count: int


class MyNested(BaseModel):
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
    for i, (_, t1) in enumerate(ds.collect()):
        assert t1 == features[i]


def test_datasets(catalog):
    ds = DataChain.datasets()
    datasets = [d for d in ds.collect("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 0

    DataChain.from_values(fib=[1, 1, 2, 3, 5, 8]).save("fibonacci")

    ds = DataChain.datasets()
    datasets = [d for d in ds.collect("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6

    ds = DataChain.datasets(object_name="foo")
    datasets = [d for d in ds.collect("foo") if d.name == "fibonacci"]
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

    ds = DataChain.from_values(fib=fib, odds=values)

    df = ds.to_pandas()
    assert len(df) == len(fib)
    assert df["fib"].tolist() == fib
    assert df["odds"].tolist() == values


def test_from_features_more_simple_types(catalog):
    ds_name = "my_ds_type"
    DataChain.from_values(
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

    ds = DataChain.from_values(file=files)

    for i, values in enumerate(ds.collect()):
        assert values[0] == files[i]


def test_gen(catalog):
    class _TestFr(BaseModel):
        file: File
        sqrt: float
        my_name: str

    ds = DataChain.from_values(t1=features)
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

    for i, (x,) in enumerate(ds.collect()):
        assert isinstance(x, _TestFr)

        fr = features[i]
        test_fr = _TestFr(file=File(name=""), sqrt=math.sqrt(fr.count), my_name=fr.nnn)
        assert x.file == test_fr.file
        assert np.isclose(x.sqrt, test_fr.sqrt)
        assert x.my_name == test_fr.my_name


def test_map(catalog):
    class _TestFr(BaseModel):
        sqrt: float
        my_name: str

    dc = DataChain.from_values(t1=features).map(
        x=lambda m_fr: _TestFr(
            sqrt=math.sqrt(m_fr.count),
            my_name=m_fr.nnn + "_suf",
        ),
        params="t1",
        output={"x": _TestFr},
    )

    x_list = list(dc.collect("x"))
    test_frs = [
        _TestFr(sqrt=math.sqrt(fr.count), my_name=fr.nnn + "_suf") for fr in features
    ]

    assert len(x_list) == len(test_frs)

    for x, test_fr in zip(x_list, test_frs):
        assert np.isclose(x.sqrt, test_fr.sqrt)
        assert x.my_name == test_fr.my_name


def test_agg(catalog):
    class _TestFr(BaseModel):
        f: File
        cnt: int
        my_name: str

    dc = DataChain.from_values(t1=features).agg(
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

    assert list(dc.collect("x")) == [
        _TestFr(
            f=File(name=""),
            cnt=sum(fr.count for fr in features if fr.nnn == "n1"),
            my_name="-".join([fr.nnn for fr in features if fr.nnn == "n1"]),
        ),
        _TestFr(
            f=File(name=""),
            cnt=sum(fr.count for fr in features if fr.nnn == "n2"),
            my_name="-".join([fr.nnn for fr in features if fr.nnn == "n2"]),
        ),
    ]


def test_agg_two_params(catalog):
    class _TestFr(BaseModel):
        f: File
        cnt: int
        my_name: str

    features2 = [
        MyFr(nnn="n1", count=6),
        MyFr(nnn="n2", count=10),
        MyFr(nnn="n1", count=2),
    ]

    ds = DataChain.from_values(t1=features, t2=features2).agg(
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

    assert list(ds.collect("x.my_name")) == ["n1-n1", "n2"]
    assert list(ds.collect("x.cnt")) == [12, 15]


def test_agg_simple_iterator(catalog):
    def func(key, val) -> Iterator[tuple[File, str]]:
        for i in range(val):
            yield File(name=""), f"{key}_{i}"

    keys = ["a", "b", "c"]
    values = [3, 1, 2]
    ds = DataChain.from_values(key=keys, val=values).gen(res=func)

    df = ds.to_pandas()
    res = df["res_1"].tolist()
    assert res == ["a_0", "a_1", "a_2", "b_0", "c_0", "c_1"]


def test_agg_simple_iterator_error(catalog):
    chain = DataChain.from_values(key=["a", "b", "c"])

    with pytest.raises(UdfSignatureError):

        def func(key) -> int:
            return 1

        chain.gen(res=func)

    with pytest.raises(UdfSignatureError):

        class _MyCls(BaseModel):
            x: int

        def func(key) -> _MyCls:  # type: ignore[misc]
            return _MyCls(x=2)

        chain.gen(res=func)

    with pytest.raises(UdfSignatureError):

        def func(key) -> tuple[File, str]:  # type: ignore[misc]
            yield None, "qq"

        chain.gen(res=func)


def test_agg_tuple_result_iterator(catalog):
    class _ImageGroup(BaseModel):
        name: str
        size: int

    def func(key, val) -> Iterator[tuple[File, _ImageGroup]]:
        n = "-".join(key)
        v = sum(val)
        yield File(name=n), _ImageGroup(name=n, size=v)

    keys = ["n1", "n2", "n1"]
    values = [1, 5, 9]
    ds = DataChain.from_values(key=keys, val=values).agg(x=func, partition_by=C("key"))

    assert list(ds.collect("x_1.name")) == ["n1-n1", "n2"]
    assert list(ds.collect("x_1.size")) == [10, 5]


def test_agg_tuple_result_generator(catalog):
    class _ImageGroup(BaseModel):
        name: str
        size: int

    def func(key, val) -> Generator[tuple[File, _ImageGroup], None, None]:
        n = "-".join(key)
        v = sum(val)
        yield File(name=n), _ImageGroup(name=n, size=v)

    keys = ["n1", "n2", "n1"]
    values = [1, 5, 9]
    ds = DataChain.from_values(key=keys, val=values).agg(x=func, partition_by=C("key"))

    assert list(ds.collect("x_1.name")) == ["n1-n1", "n2"]
    assert list(ds.collect("x_1.size")) == [10, 5]


def test_collect(catalog):
    dc = DataChain.from_values(f1=features, num=range(len(features)))

    n = 0
    for sample in dc.collect():
        assert len(sample) == 2
        fr, num = sample

        assert isinstance(fr, MyFr)
        assert isinstance(num, int)
        assert num == n
        assert fr == features[n]

        n += 1

    assert n == len(features)


def test_collect_nested_feature(catalog):
    dc = DataChain.from_values(sign1=features_nested)

    for n, sample in enumerate(dc.collect()):
        assert len(sample) == 1
        nested = sample[0]

        assert isinstance(nested, MyNested)
        assert nested == features_nested[n]


def test_select_feature(catalog):
    dc = DataChain.from_values(my_n=features_nested)

    samples = dc.select("my_n").collect()
    n = 0
    for sample in samples:
        assert sample[0] == features_nested[n]
        n += 1
    assert n == len(features_nested)

    samples = dc.select("my_n.fr").collect()
    n = 0
    for sample in samples:
        assert sample[0] == features[n]
        n += 1
    assert n == len(features_nested)

    samples = dc.select("my_n.label", "my_n.fr.count").collect()
    n = 0
    for sample in samples:
        label, count = sample
        assert label == features_nested[n].label
        assert count == features_nested[n].fr.count
        n += 1
    assert n == len(features_nested)


def test_select_columns_intersection(catalog):
    dc = DataChain.from_values(my_n=features_nested)

    samples = dc.select("my_n.fr", "my_n.fr.count").collect()
    n = 0
    for sample in samples:
        fr, count = sample
        assert fr == features_nested[n].fr
        assert count == features_nested[n].fr.count
        n += 1
    assert n == len(features_nested)


def test_select_except(catalog):
    dc = DataChain.from_values(fr1=features_nested, fr2=features)

    samples = dc.select_except("fr2").collect()
    n = 0
    for sample in samples:
        fr = sample[0]
        assert fr == features_nested[n]
        n += 1
    assert n == len(features_nested)


def test_select_wrong_type(catalog):
    dc = DataChain.from_values(fr1=features_nested, fr2=features)

    with pytest.raises(SignalResolvingTypeError):
        list(dc.select(4).collect())

    with pytest.raises(SignalResolvingTypeError):
        list(dc.select_except(features[0]).collect())


def test_select_except_error(catalog):
    dc = DataChain.from_values(fr1=features_nested, fr2=features)

    with pytest.raises(SignalResolvingError):
        list(dc.select_except("not_exist", "file").collect())

    with pytest.raises(SignalResolvingError):
        list(dc.select_except("fr1.label", "file").collect())


def test_select_restore_from_saving(catalog):
    dc = DataChain.from_values(my_n=features_nested)

    name = "test_test_select_save"
    dc.select("my_n.fr").save(name)

    restored = DataChain.from_dataset(name)
    n = 0
    restored_sorted = sorted(restored.collect(), key=lambda x: x[0].count)
    features_sorted = sorted(features, key=lambda x: x.count)
    for sample in restored_sorted:
        assert sample[0] == features_sorted[n]
        n += 1
    assert n == len(features_nested)


def test_from_dataset_name_version(catalog):
    name = "test-version"
    DataChain.from_values(
        first_name=["Alice", "Bob", "Charlie"],
        age=[40, 30, None],
        city=[
            "Houston",
            "Los Angeles",
            None,
        ],
    ).save(name)

    dc = DataChain.from_dataset(name)
    assert dc.name == name
    assert dc.version


def test_chain_of_maps(catalog):
    dc = (
        DataChain.from_values(my_n=features_nested)
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

    ds = DataChain.from_values(key=[123]).map(emd=get_vector)

    df = ds.to_pandas()
    assert np.allclose(df["emd"].tolist()[0], vector)


def test_vector_of_vectors(catalog):
    vector = [[3.14, 2.72, 1.62], [1.0, 2.0, 3.0]]

    def get_vector(key) -> list[list[float]]:
        return vector

    ds = DataChain.from_values(key=[123]).map(emd_list=get_vector)

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
        DataChain.from_values(key=[123]).map(emd=get_vector)


def test_collect_single_item(catalog):
    names = ["f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json"]
    sizes = [1, 2, 3, 4, 5]
    files = [File(name=name, size=size) for name, size in zip(names, sizes)]

    scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    chain = DataChain.from_values(file=files, score=scores)

    assert list(chain.collect("file")) == files
    assert list(chain.collect("file.name")) == names
    assert list(chain.collect("file.size")) == sizes
    assert list(chain.collect("file.source")) == [""] * len(names)
    assert np.allclose(list(chain.collect("score")), scores)

    for actual, expected in zip(
        chain.collect("file.size", "score"), [[x, y] for x, y in zip(sizes, scores)]
    ):
        assert len(actual) == 2
        assert actual[0] == expected[0]
        assert math.isclose(actual[1], expected[1], rel_tol=1e-7)


def test_default_output_type(catalog):
    names = ["f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json"]
    suffix = "-new"

    chain = DataChain.from_values(name=names).map(res1=lambda name: name + suffix)

    assert list(chain.collect("res1")) == [t + suffix for t in names]


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


def test_parse_tabular_partitions(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path, partition_cols=["first_name"])
    dc = (
        DataChain.from_storage(path.as_uri())
        .filter(C("parent").glob("*first_name=Alice*"))
        .parse_tabular(partitioning="hive")
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    assert df1.equals(df.loc[:0])


def test_parse_tabular_empty(tmp_dir, catalog):
    path = tmp_dir / "test.parquet"
    with pytest.raises(FileNotFoundError):
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


def test_parse_tabular_output_dict(tmp_dir, catalog):
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


def test_parse_tabular_output_feature(tmp_dir, catalog):
    class Output(BaseModel):
        fname: str
        age: int
        loc: str

    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    dc = DataChain.from_storage(path.as_uri()).parse_tabular(
        format="json", output=Output
    )
    df1 = dc.select("fname", "age", "loc").to_pandas()
    df.columns = ["fname", "age", "loc"]
    assert df1.equals(df)


def test_parse_tabular_output_list(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    output = ["fname", "age", "loc"]
    dc = DataChain.from_storage(path.as_uri()).parse_tabular(
        format="json", output=output
    )
    df1 = dc.select("fname", "age", "loc").to_pandas()
    df.columns = ["fname", "age", "loc"]
    assert df1.equals(df)


def test_from_csv(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path)
    dc = DataChain.from_csv(path.as_uri())
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert df1.equals(df)


def test_from_csv_no_header_error(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    with pytest.raises(DataChainParamsError):
        DataChain.from_csv(path.as_uri(), header=False)


def test_from_csv_no_header_output_dict(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    dc = DataChain.from_csv(
        path.as_uri(), header=False, output={"first_name": str, "age": int, "city": str}
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert (df1.values != df.values).sum() == 0


def test_from_csv_no_header_output_feature(tmp_dir, catalog):
    class Output(BaseModel):
        first_name: str
        age: int
        city: str

    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    dc = DataChain.from_csv(path.as_uri(), header=False, output=Output)
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert (df1.values != df.values).sum() == 0


def test_from_csv_no_header_output_list(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    dc = DataChain.from_csv(
        path.as_uri(), header=False, output=["first_name", "age", "city"]
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert (df1.values != df.values).sum() == 0


def test_from_csv_tab_delimited(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, sep="\t")
    dc = DataChain.from_csv(path.as_uri(), delimiter="\t")
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert df1.equals(df)


def test_from_parquet(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    dc = DataChain.from_parquet(path.as_uri())
    df1 = dc.select("first_name", "age", "city").to_pandas()

    assert df1.equals(df)


def test_from_parquet_partitioned(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path, partition_cols=["first_name"])
    dc = DataChain.from_parquet(path.as_uri())
    df1 = dc.select("first_name", "age", "city").to_pandas()
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    assert df1.equals(df)


def test_to_parquet(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    dc = DataChain.from_pandas(df)

    path = tmp_dir / "test.parquet"
    dc.to_parquet(path)

    assert path.is_file()
    pd.testing.assert_frame_equal(pd.read_parquet(path), df)


def test_to_parquet_partitioned(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    dc = DataChain.from_pandas(df)

    path = tmp_dir / "parquets"
    dc.to_parquet(path, partition_cols=["first_name"])

    assert set(path.iterdir()) == {
        path / f"first_name={name}" for name in df["first_name"]
    }
    df1 = pd.read_parquet(path)
    df1 = df1.reindex(columns=df.columns)
    df1["first_name"] = df1["first_name"].astype("str")
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df)


@pytest.mark.parametrize("processes", [False, 2, True])
def test_parallel(processes, catalog):
    prefix = "t & "
    vals = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    res = list(
        DataChain.from_values(key=vals)
        .settings(parallel=processes)
        .map(res=lambda key: prefix + key)
        .collect("res")
    )

    assert res == [prefix + v for v in vals]


def test_exec(catalog):
    names = ("f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json")
    all_names = set()

    dc = (
        DataChain.from_values(name=names)
        .map(nop=lambda name: all_names.add(name))
        .exec()
    )
    assert isinstance(dc, DataChain)
    assert all_names == set(names)


def test_extend_features(catalog):
    dc = DataChain.from_values(f1=features, num=range(len(features)))

    res = dc._extend_to_data_model("select", "num")
    assert isinstance(res, DataChain)
    assert res.signals_schema.values == {"num": int}

    res = dc._extend_to_data_model("sum", "num")
    assert res == sum(range(len(features)))


def test_from_storage_object_name(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    dc = DataChain.from_storage(path.as_uri(), object_name="custom")
    assert dc.schema["custom"] == File


def test_from_features_object_name(tmp_dir, catalog):
    fib = [1, 1, 2, 3, 5, 8]
    values = ["odd" if num % 2 else "even" for num in fib]

    dc = DataChain.from_values(fib=fib, odds=values, object_name="custom")
    assert "custom.fib" in dc.to_pandas(flatten=True).columns


def test_parse_tabular_object_name(tmp_dir, catalog):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    dc = DataChain.from_storage(path.as_uri()).parse_tabular(object_name="tbl")
    assert "tbl.first_name" in dc.to_pandas(flatten=True).columns


def test_sys_feature(tmp_dir, catalog):
    ds = DataChain.from_values(t1=features)
    ds_sys = ds.settings(sys=True)
    assert not ds._sys
    assert ds_sys._sys

    args = []
    ds_sys.map(res=lambda sys, t1: args.append((sys, t1))).save("ds_sys")

    sys_cls = Sys.model_construct
    assert args == [
        (sys_cls(id=1, rand=ANY), MyFr(nnn="n1", count=3)),
        (sys_cls(id=2, rand=ANY), MyFr(nnn="n2", count=5)),
        (sys_cls(id=3, rand=ANY), MyFr(nnn="n1", count=1)),
    ]
    assert "sys" not in ds_sys.catalog.get_dataset("ds_sys").feature_schema

    ds_no_sys = ds_sys.settings(sys=False)
    assert not ds_no_sys._sys

    args = []
    ds_no_sys.map(res=lambda t1: args.append(t1)).save("ds_no_sys")
    assert args == [
        MyFr(nnn="n1", count=3),
        MyFr(nnn="n2", count=5),
        MyFr(nnn="n1", count=1),
    ]
    assert "sys" not in ds_no_sys.catalog.get_dataset("ds_no_sys").feature_schema


def test_to_pandas_multi_level():
    df = DataChain.from_values(t1=features).to_pandas()

    assert "t1" in df.columns
    assert "nnn" in df["t1"].columns
    assert "count" in df["t1"].columns
    assert df["t1"]["count"].tolist() == [3, 5, 1]


def test_mutate():
    chain = DataChain.from_values(t1=features).mutate(
        circle=2 * 3.14 * Column("t1.count"), place="pref_" + Column("t1.nnn")
    )

    assert chain.signals_schema.values["circle"] is float
    assert chain.signals_schema.values["place"] is str

    expected = [fr.count * 2 * 3.14 for fr in features]
    np.testing.assert_allclose(list(chain.collect("circle")), expected)


@pytest.mark.parametrize("with_function", [True, False])
def test_order_by_with_nested_columns(with_function):
    names = ["a.txt", "c.txt", "d.txt", "a.txt", "b.txt"]

    dc = DataChain.from_values(file=[File(name=name) for name in names])
    if with_function:
        from datachain.sql.functions import rand

        dc = dc.order_by("file.name", rand())
    else:
        dc = dc.order_by("file.name")

    assert list(dc.collect("file.name")) == [
        "a.txt",
        "a.txt",
        "b.txt",
        "c.txt",
        "d.txt",
    ]


@pytest.mark.parametrize("with_function", [True, False])
def test_order_by_descending(with_function):
    names = ["a.txt", "c.txt", "d.txt", "a.txt", "b.txt"]

    dc = DataChain.from_values(file=[File(name=name) for name in names])
    if with_function:
        from datachain.sql.functions import rand

        dc = dc.order_by("file.name", rand(), descending=True)
    else:
        dc = dc.order_by("file.name", descending=True)

    assert list(dc.collect("file.name")) == [
        "d.txt",
        "c.txt",
        "b.txt",
        "a.txt",
        "a.txt",
    ]


def test_union(catalog):
    chain1 = DataChain.from_values(value=[1, 2])
    chain2 = DataChain.from_values(value=[3, 4])
    chain3 = chain1 | chain2
    assert chain3.count() == 4
    assert sorted(chain3.collect("value")) == [1, 2, 3, 4]


def test_subtract(catalog):
    chain1 = DataChain.from_values(a=[1, 1, 2], b=["x", "y", "z"])
    chain2 = DataChain.from_values(a=[1, 2], b=["x", "y"])
    assert set(chain1.subtract(chain2, on=["a", "b"]).collect()) == {(1, "y"), (2, "z")}
    assert set(chain1.subtract(chain2, on=["b"]).collect()) == {(2, "z")}
    assert set(chain1.subtract(chain2, on=["a"]).collect()) == set()
    assert set(chain1.subtract(chain2).collect()) == {(1, "y"), (2, "z")}
    assert chain1.subtract(chain1).count() == 0

    chain3 = DataChain.from_values(a=[1, 3], c=["foo", "bar"])
    assert set(chain1.subtract(chain3, on="a").collect()) == {(2, "z")}
    assert set(chain1.subtract(chain3).collect()) == {(2, "z")}


def test_subtract_error(catalog):
    chain1 = DataChain.from_values(a=[1, 1, 2], b=["x", "y", "z"])
    chain2 = DataChain.from_values(a=[1, 2], b=["x", "y"])
    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=[])
    with pytest.raises(TypeError):
        chain1.subtract(chain2, on=42)

    chain3 = DataChain.from_values(c=["foo", "bar"])
    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain3)


def test_column_math():
    fib = [1, 1, 2, 3, 5, 8]
    chain = DataChain.from_values(num=fib)

    ch = chain.mutate(add2=Column("num") + 2)
    assert list(ch.collect("add2")) == [x + 2 for x in fib]

    ch = chain.mutate(div2=Column("num") / 2.0)
    assert list(ch.collect("div2")) == [x / 2.0 for x in fib]

    ch2 = ch.mutate(x=1 - Column("div2"))
    assert list(ch2.collect("x")) == [1 - (x / 2.0) for x in fib]


def test_from_values_array_of_floats():
    embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    chain = DataChain.from_values(emd=embeddings)

    assert list(chain.collect("emd")) == embeddings
