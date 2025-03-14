import datetime
import json
import math
import os
import re
from collections.abc import Generator, Iterator
from unittest.mock import ANY

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset
from pydantic import BaseModel

from datachain import Column
from datachain.lib.data_model import DataModel
from datachain.lib.dc import C, DataChain, DatasetPrepareError, Sys
from datachain.lib.file import File
from datachain.lib.listing import LISTING_PREFIX
from datachain.lib.listing_info import ListingInfo
from datachain.lib.signal_schema import (
    SignalResolvingError,
    SignalResolvingTypeError,
    SignalSchema,
)
from datachain.lib.udf_signature import UdfSignatureError
from datachain.lib.utils import DataChainColumnError, DataChainParamsError
from datachain.sql.types import Float, Int64, String
from tests.utils import ANY_VALUE, df_equal, skip_if_not_sqlite, sort_df, sorted_dicts

DF_DATA = {
    "first_name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "age": [25, 30, 35, 40, 45],
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
}

DF_DATA_NESTED_NOT_NORMALIZED = {
    "nA-mE": [
        {"first-SELECT": "Ivan"},
        {"first-SELECT": "Alice", "l--as@t": "Smith"},
        {"l--as@t": "Jones", "first-SELECT": "Bob"},
        {"first-SELECT": "Charlie", "l--as@t": "Brown"},
        {"first-SELECT": "David", "l--as@t": "White"},
        {"first-SELECT": "Eva", "l--as@t": "Black"},
    ],
    "AgE": [41, 25, 30, 35, 40, 45],
    "citY": ["San Francisco", "New York", "Los Angeles", None, "Houston", "Phoenix"],
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


features = sorted(
    [MyFr(nnn="n1", count=3), MyFr(nnn="n2", count=5), MyFr(nnn="n1", count=1)],
    key=lambda f: (f.nnn, f.count),
)

features_nested = [
    MyNested(fr=fr, label=f"label_{num}") for num, fr in enumerate(features)
]


def sort_files(files):
    return sorted(files, key=lambda f: (f.path, f.size))


def test_pandas_conversion(test_session):
    df = pd.DataFrame(DF_DATA)
    df1 = DataChain.from_pandas(df, session=test_session)
    df1 = df1.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_pandas_conversion_in_memory():
    df = pd.DataFrame(DF_DATA)
    dc = DataChain.from_pandas(df, in_memory=True)
    assert dc.session.catalog.in_memory is True
    assert dc.session.catalog.metastore.db.db_file == ":memory:"
    assert dc.session.catalog.warehouse.db.db_file == ":memory:"
    dc = dc.select("first_name", "age", "city").to_pandas()
    assert dc.equals(df)


def test_pandas_uppercase_columns(test_session):
    data = {
        "FirstName": ["Alice", "Bob", "Charlie", "David", "Eva"],
        "Age": [25, 30, 35, 40, 45],
        "City": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
    }
    df = DataChain.from_pandas(pd.DataFrame(data), session=test_session).to_pandas()
    assert all(col not in df.columns for col in data)
    assert all(col.lower() in df.columns for col in data)


def test_pandas_incorrect_column_names(test_session):
    with pytest.raises(DataChainParamsError):
        DataChain.from_pandas(
            pd.DataFrame({"First Name": ["Alice", "Bob", "Charlie", "David", "Eva"]}),
            session=test_session,
        )

    with pytest.raises(DataChainParamsError):
        DataChain.from_pandas(
            pd.DataFrame({"": ["Alice", "Bob", "Charlie", "David", "Eva"]}),
            session=test_session,
        )

    with pytest.raises(DataChainParamsError):
        DataChain.from_pandas(
            pd.DataFrame({"First@Name": ["Alice", "Bob", "Charlie", "David", "Eva"]}),
            session=test_session,
        )


def test_from_features_basic(test_session):
    ds = DataChain.from_records(DataChain.DEFAULT_FILE_RECORD, session=test_session)
    ds = ds.gen(lambda prm: [File(path="")] * 5, params="path", output={"file": File})

    ds_name = "my_ds"
    ds.save(ds_name)
    ds = DataChain.from_dataset(name=ds_name)

    assert isinstance(ds._query.feature_schema, dict)
    assert isinstance(ds.signals_schema, SignalSchema)
    assert ds.schema.keys() == {"file"}
    assert set(ds.schema.values()) == {File}


@skip_if_not_sqlite
def test_from_features_basic_in_memory():
    ds = DataChain.from_records(DataChain.DEFAULT_FILE_RECORD, in_memory=True)
    assert ds.session.catalog.in_memory is True
    assert ds.session.catalog.metastore.db.db_file == ":memory:"
    assert ds.session.catalog.warehouse.db.db_file == ":memory:"
    ds = ds.gen(lambda prm: [File(path="")] * 5, params="path", output={"file": File})

    ds_name = "my_ds"
    ds.save(ds_name)
    ds = DataChain.from_dataset(name=ds_name)

    assert isinstance(ds._query.feature_schema, dict)
    assert isinstance(ds.signals_schema, SignalSchema)
    assert ds.schema.keys() == {"file"}
    assert set(ds.schema.values()) == {File}


def test_from_features(test_session):
    ds = DataChain.from_records(DataChain.DEFAULT_FILE_RECORD, session=test_session)
    ds = ds.gen(
        lambda prm: list(zip([File(path="")] * len(features), features)),
        params="path",
        output={"file": File, "t1": MyFr},
    )

    assert [r[1] for r in ds.order_by("t1.nnn", "t1.count").collect()] == features


def test_from_records_empty_chain_with_schema(test_session):
    schema = {"my_file": File, "my_col": int}
    ds = DataChain.from_records([], schema=schema, session=test_session)

    ds_name = "my_ds"
    ds.save(ds_name)
    ds = DataChain.from_dataset(name=ds_name)

    assert isinstance(ds._query.feature_schema, dict)
    assert isinstance(ds.signals_schema, SignalSchema)
    assert ds.schema.keys() == {"my_file", "my_col"}
    assert set(ds.schema.values()) == {File, int}
    assert ds.count() == 0

    # check that columns have actually been created from schema
    catalog = test_session.catalog
    dr = catalog.warehouse.dataset_rows(catalog.get_dataset(ds_name))
    assert sorted([c.name for c in dr.columns]) == sorted(
        ds.signals_schema.db_signals()
    )


def test_from_records_empty_chain_without_schema(test_session):
    ds = DataChain.from_records([], schema=None, session=test_session)

    ds_name = "my_ds"
    ds.save(ds_name)
    ds = DataChain.from_dataset(name=ds_name)

    assert ds.schema.keys() == {
        "source",
        "path",
        "size",
        "version",
        "etag",
        "is_latest",
        "last_modified",
        "location",
    }
    assert ds.count() == 0

    # check that columns have actually been created from schema
    catalog = test_session.catalog
    dr = catalog.warehouse.dataset_rows(catalog.get_dataset(ds_name))
    assert sorted([c.name for c in dr.columns]) == sorted(
        ds.signals_schema.db_signals()
    )


def test_datasets(test_session):
    ds = DataChain.datasets(session=test_session)
    datasets = [d for d in ds.collect("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 0

    DataChain.from_values(fib=[1, 1, 2, 3, 5, 8], session=test_session).save(
        "fibonacci"
    )

    ds = DataChain.datasets(session=test_session)
    datasets = [d for d in ds.collect("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6

    ds = DataChain.datasets(object_name="foo", session=test_session)
    datasets = [d for d in ds.collect("foo") if d.name == "fibonacci"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6


def test_datasets_studio(studio_datasets, test_session):
    DataChain.from_values(fib=[1, 1, 2, 3, 5, 8], session=test_session).save(
        "fibonacci"
    )
    ds = DataChain.datasets(studio=True, session=test_session)
    # Local datasets are not included in the list
    datasets = [d for d in ds.collect("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 0

    # Studio datasets are included in the list
    datasets = [d for d in ds.collect("dataset") if d.name == "cats"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6

    # Exclude studio datasets
    ds = DataChain.datasets(studio=False, session=test_session)
    datasets = [d for d in ds.collect("dataset") if d.name == "cats"]
    assert len(datasets) == 0


@skip_if_not_sqlite
def test_datasets_in_memory():
    ds = DataChain.datasets(in_memory=True)
    assert ds.session.catalog.in_memory is True
    assert ds.session.catalog.metastore.db.db_file == ":memory:"
    assert ds.session.catalog.warehouse.db.db_file == ":memory:"
    datasets = [d for d in ds.collect("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 0

    DataChain.from_values(fib=[1, 1, 2, 3, 5, 8]).save("fibonacci")

    ds = DataChain.datasets()
    assert ds.session.catalog.in_memory is True
    assert ds.session.catalog.metastore.db.db_file == ":memory:"
    assert ds.session.catalog.warehouse.db.db_file == ":memory:"
    datasets = [d for d in ds.collect("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6

    ds = DataChain.datasets(object_name="foo")
    assert ds.session.catalog.in_memory is True
    assert ds.session.catalog.metastore.db.db_file == ":memory:"
    assert ds.session.catalog.warehouse.db.db_file == ":memory:"
    datasets = [d for d in ds.collect("foo") if d.name == "fibonacci"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6


def test_listings(test_session, tmp_dir):
    df = pd.DataFrame(DF_DATA)
    df.to_parquet(tmp_dir / "df.parquet")

    uri = tmp_dir.as_uri()
    DataChain.from_storage(uri, session=test_session)

    # check that listing is not returned as normal dataset
    assert not any(
        n.startswith(LISTING_PREFIX)
        for n in [
            ds.name
            for ds in DataChain.datasets(session=test_session).collect("dataset")
        ]
    )

    listings = list(DataChain.listings(session=test_session).collect("listing"))
    assert len(listings) == 1
    listing = listings[0]
    assert isinstance(listing, ListingInfo)
    assert listing.storage_uri == uri
    assert listing.is_expired is False
    assert listing.expires
    assert listing.version == 1
    assert listing.num_objects == 1
    # Exact number if unreliable here since it depends on the PyArrow version
    assert listing.size > 1000
    assert listing.size < 5000
    assert listing.status == 4


def test_listings_reindex(test_session, tmp_dir):
    df = pd.DataFrame(DF_DATA)
    df.to_parquet(tmp_dir / "df.parquet")

    uri = tmp_dir.as_uri()

    DataChain.from_storage(uri, session=test_session)
    assert len(list(DataChain.listings(session=test_session).collect("listing"))) == 1

    DataChain.from_storage(uri, session=test_session)
    assert len(list(DataChain.listings(session=test_session).collect("listing"))) == 1

    DataChain.from_storage(uri, session=test_session, update=True)
    listings = list(DataChain.listings(session=test_session).collect("listing"))
    assert len(listings) == 2
    listings.sort(key=lambda lst: lst.version)
    assert listings[0].storage_uri == uri
    assert listings[0].version == 1
    assert listings[1].storage_uri == uri
    assert listings[1].version == 2


def test_listings_reindex_subpath_local_file_system(test_session, tmp_dir):
    subdir = tmp_dir / "subdir"
    os.mkdir(subdir)

    df = pd.DataFrame(DF_DATA)
    df.to_parquet(tmp_dir / "df.parquet")
    df.to_parquet(tmp_dir / "df2.parquet")
    df.to_parquet(subdir / "df3.parquet")

    assert DataChain.from_storage(tmp_dir.as_uri(), session=test_session).count() == 3
    assert DataChain.from_storage(subdir.as_uri(), session=test_session).count() == 1


def test_preserve_feature_schema(test_session):
    ds = DataChain.from_records(DataChain.DEFAULT_FILE_RECORD, session=test_session)
    ds = ds.gen(
        lambda prm: list(zip([File(path="")] * len(features), features, features)),
        params="path",
        output={"file": File, "t1": MyFr, "t2": MyFr},
    )

    ds_name = "my_ds1"
    ds.save(ds_name)
    ds = DataChain.from_dataset(name=ds_name)

    assert isinstance(ds._query.feature_schema, dict)
    assert isinstance(ds.signals_schema, SignalSchema)
    assert ds.schema.keys() == {"t1", "t2", "file"}
    assert set(ds.schema.values()) == {MyFr, File}


def test_from_features_simple_types(test_session):
    fib = [1, 1, 2, 3, 5, 8]
    values = ["odd" if num % 2 else "even" for num in fib]

    ds = DataChain.from_values(fib=fib, odds=values, session=test_session)

    df = sort_df(ds.to_pandas())
    assert len(df) == len(fib)
    assert df["fib"].tolist() == fib
    assert df["odds"].tolist() == values


@skip_if_not_sqlite
def test_from_features_simple_types_in_memory():
    fib = [1, 1, 2, 3, 5, 8]
    values = ["odd" if num % 2 else "even" for num in fib]

    ds = DataChain.from_values(fib=fib, odds=values, in_memory=True)
    assert ds.session.catalog.in_memory is True
    assert ds.session.catalog.metastore.db.db_file == ":memory:"
    assert ds.session.catalog.warehouse.db.db_file == ":memory:"

    df = ds.to_pandas()
    assert len(df) == len(fib)
    assert df["fib"].tolist() == fib
    assert df["odds"].tolist() == values


def test_from_features_more_simple_types(test_session):
    ds = DataChain.from_values(
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
        session=test_session,
    )

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


def test_file_list(test_session):
    names = ["f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json"]
    sizes = [1, 2, 3, 4, 5]
    files = [File(path=name, size=size) for name, size in zip(names, sizes)]

    ds = DataChain.from_values(file=files, session=test_session)

    assert sort_files(files) == [
        r[0] for r in ds.order_by("file.path", "file.size").collect()
    ]


def test_gen(test_session):
    class _TestFr(BaseModel):
        file: File
        sqrt: float
        my_name: str

    ds = DataChain.from_values(t1=features, session=test_session)
    ds = ds.gen(
        x=lambda m_fr: [
            _TestFr(
                file=File(path=""),
                sqrt=math.sqrt(m_fr.count),
                my_name=m_fr.nnn,
            )
        ],
        params="t1",
        output={"x": _TestFr},
    )

    for i, (x,) in enumerate(ds.order_by("x.my_name", "x.sqrt").collect()):
        assert isinstance(x, _TestFr)

        fr = features[i]
        test_fr = _TestFr(file=File(path=""), sqrt=math.sqrt(fr.count), my_name=fr.nnn)
        assert x.file == test_fr.file
        assert np.isclose(x.sqrt, test_fr.sqrt)
        assert x.my_name == test_fr.my_name


def test_map(test_session):
    class _TestFr(BaseModel):
        sqrt: float
        my_name: str

    dc = DataChain.from_values(t1=features, session=test_session).map(
        x=lambda m_fr: _TestFr(
            sqrt=math.sqrt(m_fr.count),
            my_name=m_fr.nnn + "_suf",
        ),
        params="t1",
        output={"x": _TestFr},
    )

    x_list = list(dc.order_by("x.my_name", "x.sqrt").collect("x"))
    test_frs = [
        _TestFr(sqrt=math.sqrt(fr.count), my_name=fr.nnn + "_suf") for fr in features
    ]

    assert len(x_list) == len(test_frs)

    for x, test_fr in zip(x_list, test_frs):
        assert np.isclose(x.sqrt, test_fr.sqrt)
        assert x.my_name == test_fr.my_name


def test_map_existing_column_after_step(test_session):
    dc = DataChain.from_values(t1=features, session=test_session).map(
        x=lambda _: "test",
        params="t1",
        output={"x": str},
    )

    with pytest.raises(ValueError):
        dc.map(
            x=lambda _: "test",
            params="t1",
            output={"x": str},
        ).exec()


def test_agg(test_session):
    class _TestFr(BaseModel):
        f: File
        cnt: int
        my_name: str

    dc = DataChain.from_values(t1=features, session=test_session).agg(
        x=lambda frs: [
            _TestFr(
                f=File(path=""),
                cnt=sum(f.count for f in frs),
                my_name="-".join([fr.nnn for fr in frs]),
            )
        ],
        partition_by=C.t1.nnn,
        params="t1",
        output={"x": _TestFr},
    )

    assert list(dc.order_by("x.my_name").collect("x")) == [
        _TestFr(
            f=File(path=""),
            cnt=sum(fr.count for fr in features if fr.nnn == "n1"),
            my_name="-".join([fr.nnn for fr in features if fr.nnn == "n1"]),
        ),
        _TestFr(
            f=File(path=""),
            cnt=sum(fr.count for fr in features if fr.nnn == "n2"),
            my_name="-".join([fr.nnn for fr in features if fr.nnn == "n2"]),
        ),
    ]


def test_agg_two_params(test_session):
    class _TestFr(BaseModel):
        f: File
        cnt: int
        my_name: str

    features2 = [
        MyFr(nnn="n1", count=6),
        MyFr(nnn="n2", count=10),
        MyFr(nnn="n1", count=2),
    ]

    ds = (
        DataChain.from_values(t1=features, t2=features2, session=test_session)
        .order_by("t1.nnn", "t2.nnn")
        .agg(
            x=lambda frs1, frs2: [
                _TestFr(
                    f=File(path=""),
                    cnt=sum(f1.count + f2.count for f1, f2 in zip(frs1, frs2)),
                    my_name="-".join([fr.nnn for fr in frs1]),
                )
            ],
            partition_by=C.t1.nnn,
            params=("t1", "t2"),
            output={"x": _TestFr},
        )
    )

    assert list(ds.order_by("x.my_name").collect("x.my_name")) == ["n1-n1", "n2"]
    assert list(ds.order_by("x.cnt").collect("x.cnt")) == [7, 20]


def test_agg_simple_iiterator(test_session):
    def func(key, val) -> Iterator[tuple[File, str]]:
        for i in range(val):
            yield File(path=""), f"{key}_{i}"

    keys = ["a", "b", "c"]
    values = [3, 1, 2]
    ds = DataChain.from_values(key=keys, val=values, session=test_session).gen(res=func)

    df = sort_df(ds.to_pandas())
    res = df["res_1"].tolist()
    assert res == ["a_0", "a_1", "a_2", "b_0", "c_0", "c_1"]


def test_agg_simple_iterator_error(test_session):
    chain = DataChain.from_values(key=["a", "b", "c"], session=test_session)

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


def test_agg_tuple_result_iterator(test_session):
    class _ImageGroup(BaseModel):
        name: str
        size: int

    def func(key, val) -> Iterator[tuple[File, _ImageGroup]]:
        n = "-".join(key)
        v = sum(val)
        yield File(path=n), _ImageGroup(name=n, size=v)

    keys = ["n1", "n2", "n1"]
    values = [1, 5, 9]
    ds = DataChain.from_values(key=keys, val=values, session=test_session).agg(
        x=func, partition_by=C("key")
    )

    assert list(ds.order_by("x_1.name").collect("x_1.name")) == ["n1-n1", "n2"]
    assert list(ds.order_by("x_1.size").collect("x_1.size")) == [5, 10]


def test_agg_tuple_result_generator(test_session):
    class _ImageGroup(BaseModel):
        name: str
        size: int

    def func(key, val) -> Generator[tuple[File, _ImageGroup], None, None]:
        n = "-".join(key)
        v = sum(val)
        yield File(path=n), _ImageGroup(name=n, size=v)

    keys = ["n1", "n2", "n1"]
    values = [1, 5, 9]
    ds = (
        DataChain.from_values(key=keys, val=values, session=test_session)
        .agg(x=func, partition_by=C("key"))
        .order_by("x_1.name")
    )

    assert list(ds.order_by("x_1.name").collect("x_1.name")) == ["n1-n1", "n2"]
    assert list(ds.order_by("x_1.size").collect("x_1.size")) == [5, 10]


def test_batch_map(test_session):
    class _TestFr(BaseModel):
        sqrt: float
        my_name: str

    dc = DataChain.from_values(t1=features, session=test_session).batch_map(
        x=lambda m_frs: [
            _TestFr(
                sqrt=math.sqrt(m_fr.count),
                my_name=m_fr.nnn + "_suf",
            )
            for m_fr in m_frs
        ],
        params="t1",
        output={"x": _TestFr},
    )

    x_list = list(dc.order_by("x.my_name", "x.sqrt").collect("x"))
    test_frs = [
        _TestFr(sqrt=math.sqrt(fr.count), my_name=fr.nnn + "_suf") for fr in features
    ]

    assert len(x_list) == len(test_frs)

    for x, test_fr in zip(x_list, test_frs):
        assert np.isclose(x.sqrt, test_fr.sqrt)
        assert x.my_name == test_fr.my_name


def test_batch_map_wrong_size(test_session):
    class _TestFr(BaseModel):
        total: int
        names: str

    dc = DataChain.from_values(t1=features, session=test_session).batch_map(
        x=lambda m_frs: [
            _TestFr(
                total=sum(m_fr.count for m_fr in m_frs),
                names="-".join([m_fr.nnn for m_fr in m_frs]),
            )
        ],
        params="t1",
        output={"x": _TestFr},
    )

    with pytest.raises(AssertionError):
        list(dc.collect())


def test_batch_map_two_params(test_session):
    class _TestFr(BaseModel):
        f: File
        cnt: int
        my_name: str

    features2 = [
        MyFr(nnn="n1", count=6),
        MyFr(nnn="n2", count=10),
        MyFr(nnn="n1", count=2),
    ]

    ds = DataChain.from_values(
        t1=features, t2=features2, session=test_session
    ).batch_map(
        x=lambda frs1, frs2: [
            _TestFr(
                f=File(path=""),
                cnt=f1.count + f2.count,
                my_name=f"{f1.nnn}-{f2.nnn}",
            )
            for f1, f2 in zip(frs1, frs2)
        ],
        params=("t1", "t2"),
        output={"x": _TestFr},
    )

    assert list(ds.order_by("x.my_name").collect("x.my_name")) == [
        "n1-n1",
        "n1-n2",
        "n2-n1",
    ]
    assert list(ds.order_by("x.cnt").collect("x.cnt")) == [7, 7, 13]


def test_batch_map_tuple_result_iterator(test_session):
    def sqrt(t1: list[int]) -> Iterator[float]:
        for val in t1:
            yield math.sqrt(val)

    dc = DataChain.from_values(t1=[1, 4, 9], session=test_session).batch_map(x=sqrt)

    assert list(dc.order_by("x").collect("x")) == [1, 2, 3]


def test_collect(test_session):
    dc = DataChain.from_values(
        f1=features, num=range(len(features)), session=test_session
    )

    n = 0
    for sample in dc.order_by("f1.nnn", "f1.count").collect():
        assert len(sample) == 2
        fr, num = sample

        assert isinstance(fr, MyFr)
        assert isinstance(num, int)
        assert num == n
        assert fr == features[n]

        n += 1

    assert n == len(features)


def test_collect_nested_feature(test_session):
    dc = DataChain.from_values(sign1=features_nested, session=test_session)

    for n, sample in enumerate(dc.order_by("sign1.fr.nnn", "sign1.fr.count").collect()):
        assert len(sample) == 1
        nested = sample[0]

        assert isinstance(nested, MyNested)
        assert nested == features_nested[n]


def test_select_from_dataset_without_sys_columns(test_session):
    from datachain import func

    dc = (
        DataChain.from_values(
            name=["a", "a", "b", "b", "b", "c"],
            val=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .group_by(cnt=func.count(), partition_by="name")
        .order_by("name")
        .select("name", "cnt")
    )

    assert dc.to_records() == [
        {"name": "a", "cnt": 2},
        {"name": "b", "cnt": 3},
        {"name": "c", "cnt": 1},
    ]


def test_select_feature(test_session):
    dc = DataChain.from_values(my_n=features_nested, session=test_session)
    dc_ordered = dc.order_by("my_n.fr.nnn", "my_n.fr.count")

    samples = dc_ordered.select("my_n").collect()
    n = 0
    for sample in samples:
        assert sample[0] == features_nested[n]
        n += 1
    assert n == len(features_nested)

    samples = dc_ordered.select("my_n.fr").collect()
    n = 0
    for sample in samples:
        assert sample[0] == features[n]
        n += 1
    assert n == len(features_nested)

    samples = dc_ordered.select("my_n.label", "my_n.fr.count").collect()
    n = 0
    for sample in samples:
        label, count = sample
        assert label == features_nested[n].label
        assert count == features_nested[n].fr.count
        n += 1
    assert n == len(features_nested)


def test_select_columns_intersection(test_session):
    dc = DataChain.from_values(my_n=features_nested, session=test_session)

    samples = (
        dc.order_by("my_n.fr.nnn", "my_n.fr.count")
        .select("my_n.fr", "my_n.fr.count")
        .collect()
    )
    n = 0
    for sample in samples:
        fr, count = sample
        assert fr == features_nested[n].fr
        assert count == features_nested[n].fr.count
        n += 1
    assert n == len(features_nested)


def test_select_except(test_session):
    dc = DataChain.from_values(fr1=features_nested, fr2=features, session=test_session)

    samples = dc.order_by("fr1.fr.nnn", "fr1.fr.count").select_except("fr2").collect()
    n = 0
    for sample in samples:
        fr = sample[0]
        assert fr == features_nested[n]
        n += 1
    assert n == len(features_nested)


def test_select_wrong_type(test_session):
    dc = DataChain.from_values(fr1=features_nested, fr2=features, session=test_session)

    with pytest.raises(SignalResolvingTypeError):
        list(dc.select(4).collect())

    with pytest.raises(SignalResolvingTypeError):
        list(dc.select_except(features[0]).collect())


def test_select_except_error(test_session):
    dc = DataChain.from_values(fr1=features_nested, fr2=features, session=test_session)

    with pytest.raises(SignalResolvingError):
        list(dc.select_except("not_exist", "file").collect())

    with pytest.raises(SignalResolvingError):
        list(dc.select_except("fr1.label", "file").collect())


def test_select_restore_from_saving(test_session):
    dc = DataChain.from_values(my_n=features_nested, session=test_session)

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


def test_select_distinct(test_session):
    class Embedding(BaseModel):
        id: int
        filename: str
        values: list[float]

    expected = [
        [0.1, 0.3],
        [0.1, 0.4],
        [0.1, 0.5],
        [0.1, 0.6],
    ]

    actual = (
        DataChain.from_values(
            embedding=[
                Embedding(id=1, filename="a.jpg", values=expected[0]),
                Embedding(id=2, filename="b.jpg", values=expected[2]),
                Embedding(id=3, filename="c.jpg", values=expected[1]),
                Embedding(id=4, filename="d.jpg", values=expected[1]),
                Embedding(id=5, filename="e.jpg", values=expected[3]),
            ],
            session=test_session,
        )
        .select("embedding.values", "embedding.filename")
        .distinct("embedding.values")
        .order_by("embedding.values")
        .collect()
    )

    actual = [emb[0] for emb in actual]
    assert len(actual) == 4
    for i in [0, 1]:
        assert np.allclose([emb[i] for emb in actual], [emp[i] for emp in expected])


def test_from_dataset_name_version(test_session):
    name = "test-version"
    DataChain.from_values(
        first_name=["Alice", "Bob", "Charlie"],
        age=[40, 30, None],
        city=[
            "Houston",
            "Los Angeles",
            None,
        ],
        session=test_session,
    ).save(name)

    dc = DataChain.from_dataset(name)
    assert dc.name == name
    assert dc.version


def test_chain_of_maps(test_session):
    dc = (
        DataChain.from_values(my_n=features_nested, session=test_session)
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


def test_vector(test_session):
    vector = [3.14, 2.72, 1.62]

    def get_vector(key) -> list[float]:
        return vector

    ds = DataChain.from_values(key=[123], session=test_session).map(emd=get_vector)

    df = ds.to_pandas()
    assert np.allclose(df["emd"].tolist()[0], vector)


def test_vector_of_vectors(test_session):
    vector = [[3.14, 2.72, 1.62], [1.0, 2.0, 3.0]]

    def get_vector(key) -> list[list[float]]:
        return vector

    ds = DataChain.from_values(key=[123], session=test_session).map(emd_list=get_vector)

    df = ds.to_pandas()
    actual = df["emd_list"].tolist()[0]
    assert len(actual) == 2
    assert np.allclose(actual[0], vector[0])
    assert np.allclose(actual[1], vector[1])


def test_unsupported_output_type(test_session):
    vector = [3.14, 2.72, 1.62]

    def get_vector(key) -> list[np.float64]:
        return [vector]

    with pytest.raises(TypeError):
        DataChain.from_values(key=[123], session=test_session).map(emd=get_vector)


def test_collect_single_item(test_session):
    names = ["f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json"]
    sizes = [1, 2, 3, 4, 5]
    files = sort_files([File(path=name, size=size) for name, size in zip(names, sizes)])

    scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    chain = DataChain.from_values(file=files, score=scores, session=test_session)
    chain = chain.order_by("file.path", "file.size")

    assert list(chain.collect("file")) == files
    assert list(chain.collect("file.path")) == names
    assert list(chain.collect("file.size")) == sizes
    assert list(chain.collect("file.source")) == [""] * len(names)
    assert np.allclose(list(chain.collect("score")), scores)

    for actual, expected in zip(
        chain.collect("file.size", "score"), [[x, y] for x, y in zip(sizes, scores)]
    ):
        assert len(actual) == 2
        assert actual[0] == expected[0]
        assert math.isclose(actual[1], expected[1], rel_tol=1e-7)


def test_default_output_type(test_session):
    names = sorted(["f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json"])
    suffix = "-new"

    chain = DataChain.from_values(name=names, session=test_session).map(
        res1=lambda name: name + suffix
    )

    assert list(chain.order_by("name").collect("res1")) == [t + suffix for t in names]


def test_parse_tabular(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    dc = DataChain.from_storage(path.as_uri(), session=test_session).parse_tabular()
    df1 = dc.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_parse_tabular_in_memory(tmp_dir):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    dc = DataChain.from_storage(path.as_uri(), in_memory=True).parse_tabular()
    assert dc.session.catalog.in_memory is True
    assert dc.session.catalog.metastore.db.db_file == ":memory:"
    assert dc.session.catalog.warehouse.db.db_file == ":memory:"
    df1 = dc.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df)


def test_parse_tabular_format(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    dc = DataChain.from_storage(path.as_uri(), session=test_session).parse_tabular(
        format="json"
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


def test_parse_nested_json(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA_NESTED_NOT_NORMALIZED)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    dc = DataChain.from_storage(path.as_uri(), session=test_session).parse_tabular(
        format="json"
    )
    # Field names are normalized, values are preserved
    # E.g. nAmE -> name, l--as@t -> l_as_t, etc
    df1 = dc.select("na_me", "age", "city").to_pandas()

    # In CH we replace None with '' for peforance reasons,
    # have to handle it here
    string_default = String.default_value(test_session.catalog.warehouse.db.dialect)

    assert sorted(df1["na_me"]["first_select"].to_list()) == sorted(
        d["first-SELECT"] for d in df["nA-mE"].to_list()
    )
    assert sorted(
        df1["na_me"]["l_as_t"].to_list(), key=lambda x: (x is None, x)
    ) == sorted(
        [d.get("l--as@t", string_default) for d in df["nA-mE"].to_list()],
        key=lambda x: (x is None, x),
    )


def test_parse_tabular_partitions(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path, partition_cols=["first_name"])
    dc = (
        DataChain.from_storage(path.as_uri(), session=test_session)
        .filter(C("file.path").glob("*first_name=Alice*"))
        .parse_tabular(partitioning="hive")
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    assert df_equal(df1, df.loc[:0])


def test_parse_tabular_no_files(test_session):
    dc = DataChain.from_values(
        f1=features, num=range(len(features)), session=test_session, in_memory=True
    )
    with pytest.raises(DatasetPrepareError):
        dc.parse_tabular()

    schema = {"file": File, "my_col": int}
    dc = DataChain.from_records([], schema=schema, session=test_session, in_memory=True)

    with pytest.raises(DatasetPrepareError):
        dc.parse_tabular()


def test_parse_tabular_unify_schema(tmp_dir, test_session):
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
        DataChain.from_storage(tmp_dir.as_uri(), session=test_session)
        .filter(C("file.path").glob("*.parquet"))
        .parse_tabular()
    )
    df = dc.select("first_name", "age", "city", "last_name", "country").to_pandas()
    df = (
        df.replace({"": None, 0: None, np.nan: None})
        .sort_values("first_name")
        .reset_index(drop=True)
    )
    assert df_equal(df, df_combined)


def test_parse_tabular_output_dict(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    output = {"fname": str, "age": int, "loc": str}
    dc = DataChain.from_storage(path.as_uri(), session=test_session).parse_tabular(
        format="json", output=output
    )
    df1 = dc.select("fname", "age", "loc").to_pandas()
    df.columns = ["fname", "age", "loc"]
    assert df_equal(df1, df)


def test_parse_tabular_output_feature(tmp_dir, test_session):
    class Output(BaseModel):
        fname: str
        age: int
        loc: str

    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    dc = DataChain.from_storage(path.as_uri(), session=test_session).parse_tabular(
        format="json", output=Output
    )
    df1 = dc.select("fname", "age", "loc").to_pandas()
    df.columns = ["fname", "age", "loc"]
    assert df_equal(df1, df)


def test_parse_tabular_output_list(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    output = ["fname", "age", "loc"]
    dc = DataChain.from_storage(path.as_uri(), session=test_session).parse_tabular(
        format="json", output=output
    )
    df1 = dc.select("fname", "age", "loc").to_pandas()
    df.columns = ["fname", "age", "loc"]
    assert df_equal(df1, df)


def test_parse_tabular_nrows(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_json(path, orient="records", lines=True)
    dc = DataChain.from_storage(path.as_uri(), session=test_session).parse_tabular(
        nrows=2, format="json"
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df[:2])


def test_parse_tabular_nrows_invalid(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    with pytest.raises(DataChainParamsError):
        DataChain.from_storage(path.as_uri(), session=test_session).parse_tabular(
            nrows=2
        )


def test_from_csv(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    dc = DataChain.from_csv(path.as_uri(), session=test_session)
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


def test_to_from_csv(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    dc_to = DataChain.from_pandas(df, session=test_session)
    path = tmp_dir / "test.csv"
    dc_to.to_csv(path)
    dc_from = DataChain.from_csv(path.as_uri(), session=test_session)
    df1 = dc_from.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_from_csv_in_memory(tmp_dir):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    dc = DataChain.from_csv(path.as_uri(), in_memory=True)
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_to_from_csv_in_memory(tmp_dir):
    df = pd.DataFrame(DF_DATA)
    dc_to = DataChain.from_pandas(df, in_memory=True)
    path = tmp_dir / "test.csv"
    dc_to.to_csv(path)
    dc_from = DataChain.from_csv(path.as_uri(), in_memory=True)
    df1 = dc_from.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


def test_from_csv_no_header_error(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    with pytest.raises(DataChainParamsError):
        DataChain.from_csv(path.as_uri(), header=False, session=test_session)


def test_from_csv_no_header_output_dict(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    dc = DataChain.from_csv(
        path.as_uri(),
        header=False,
        output={"first_name": str, "age": int, "city": str},
        session=test_session,
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert (sort_df(df1).values != sort_df(df).values).sum() == 0


def test_from_csv_no_header_output_feature(tmp_dir, test_session):
    class Output(BaseModel):
        first_name: str
        age: int
        city: str

    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    dc = DataChain.from_csv(
        path.as_uri(), header=False, output=Output, session=test_session
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert (sort_df(df1).values != sort_df(df).values).sum() == 0


def test_from_csv_no_header_output_list(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    dc = DataChain.from_csv(
        path.as_uri(),
        header=False,
        output=["first_name", "age", "city"],
        session=test_session,
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert (sort_df(df1).values != sort_df(df).values).sum() == 0


def test_from_csv_tab_delimited(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, sep="\t", index=False)
    dc = DataChain.from_csv(path.as_uri(), delimiter="\t", session=test_session)
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_from_csv_null_collect(tmp_dir, test_session):
    # Clickhouse requires setting type to Nullable(Type).
    # See https://github.com/xzkostyan/clickhouse-sqlalchemy/issues/189.
    df = pd.DataFrame(DF_DATA)
    height = [70, 65, None, 72, 68]
    gender = ["f", "m", None, "m", "f"]
    df["height"] = height
    df["gender"] = gender
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    dc = DataChain.from_csv(path.as_uri(), object_name="csv", session=test_session)
    for i, row in enumerate(dc.collect()):
        # None value in numeric column will get converted to nan.
        if not height[i]:
            assert math.isnan(row[1].height)
        else:
            assert row[1].height == height[i]
        assert row[1].gender == gender[i]


def test_from_csv_nrows(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    dc = DataChain.from_csv(path.as_uri(), nrows=2, session=test_session)
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df[:2])


def test_from_csv_column_types(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    dc = DataChain.from_csv(
        path.as_uri(), column_types={"age": "str"}, session=test_session
    )
    df1 = dc.select("first_name", "age", "city").to_pandas()
    assert df1["age"].dtype == pd.StringDtype


def test_from_csv_parse_options(tmp_dir, test_session):
    def skip_comment(row):
        if row.text.startswith("# "):
            return "skip"
        return "error"

    s = (
        "animals;n_legs;entry\n"
        "Flamingo;2;2022-03-01\n"
        "# Comment here:\n"
        "Horse;4;2022-03-02\n"
        "Brittle stars;5;2022-03-03\n"
        "Centipede;100;2022-03-04"
    )

    path = tmp_dir / "test.csv"
    path.write_text(s)

    dc = DataChain.from_csv(
        path.as_uri(),
        session=test_session,
        parse_options={"invalid_row_handler": skip_comment, "delimiter": ";"},
    )

    df = dc.select("animals", "n_legs", "entry").to_pandas()
    assert set(df["animals"]) == {"Horse", "Centipede", "Brittle stars", "Flamingo"}


def test_to_csv_features(tmp_dir, test_session):
    dc_to = DataChain.from_values(
        f1=features, num=range(len(features)), session=test_session
    )
    path = tmp_dir / "test.csv"
    dc_to.order_by("f1.nnn", "f1.count").to_csv(path)
    with open(path) as f:
        lines = f.read().split("\n")
    assert lines == ["f1.nnn,f1.count,num", "n1,1,0", "n1,3,1", "n2,5,2", ""]


def test_to_tsv_features(tmp_dir, test_session):
    dc_to = DataChain.from_values(
        f1=features, num=range(len(features)), session=test_session
    )
    path = tmp_dir / "test.csv"
    dc_to.order_by("f1.nnn", "f1.count").to_csv(path, delimiter="\t")
    with open(path) as f:
        lines = f.read().split("\n")
    assert lines == ["f1.nnn\tf1.count\tnum", "n1\t1\t0", "n1\t3\t1", "n2\t5\t2", ""]


def test_to_csv_features_nested(tmp_dir, test_session):
    dc_to = DataChain.from_values(sign1=features_nested, session=test_session)
    path = tmp_dir / "test.csv"
    dc_to.order_by("sign1.fr.nnn", "sign1.fr.count").to_csv(path)
    with open(path) as f:
        lines = f.read().split("\n")
    assert lines == [
        "sign1.label,sign1.fr.nnn,sign1.fr.count",
        "label_0,n1,1",
        "label_1,n1,3",
        "label_2,n2,5",
        "",
    ]


@pytest.mark.parametrize("column_type", (str, dict))
@pytest.mark.parametrize("object_name", (None, "test_object_name"))
@pytest.mark.parametrize("model_name", (None, "TestModelNameExploded"))
def test_explode(tmp_dir, test_session, column_type, object_name, model_name):
    df = pd.DataFrame(DF_DATA_NESTED_NOT_NORMALIZED)
    path = tmp_dir / "test.json"
    df.to_json(path, orient="records", lines=True)

    dc = (
        DataChain.from_storage(path.as_uri(), session=test_session)
        .gen(
            content=lambda file: (ln for ln in file.read_text().split("\n") if ln),
            output=column_type,
        )
        .explode(
            "content",
            object_name=object_name,
            model_name=model_name,
            schema_sample_size=2,
        )
    )

    object_name = object_name or "content_expl"
    model_name = model_name or "ContentExplodedModel"

    # In CH we have (atm at least) None converted to ''
    # for performance reasons, so we need to handle this case
    string_default = String.default_value(test_session.catalog.warehouse.db.dialect)

    assert set(
        dc.collect(
            f"{object_name}.na_me.first_select",
            f"{object_name}.age",
            f"{object_name}.city",
        )
    ) == {
        ("Alice", 25, "New York"),
        ("Bob", 30, "Los Angeles"),
        ("Charlie", 35, string_default),
        ("David", 40, "Houston"),
        ("Eva", 45, "Phoenix"),
        ("Ivan", 41, "San Francisco"),
    }

    assert next(dc.limit(1).collect(object_name)).__class__.__name__ == model_name


def test_explode_raises_on_wrong_column_type(test_session):
    dc = DataChain.from_values(f1=features, session=test_session)

    with pytest.raises(TypeError):
        dc.explode("f1.count")


def test_to_json_features(tmp_dir, test_session):
    dc_to = DataChain.from_values(
        f1=features, num=range(len(features)), session=test_session
    )
    path = tmp_dir / "test.json"
    dc_to.order_by("f1.nnn", "f1.count").to_json(path)
    with open(path) as f:
        values = json.load(f)
    assert values == [
        {"f1": {"nnn": f.nnn, "count": f.count}, "num": n}
        for n, f in enumerate(features)
    ]


def test_to_json_features_nested(tmp_dir, test_session):
    dc_to = DataChain.from_values(sign1=features_nested, session=test_session)
    path = tmp_dir / "test.json"
    dc_to.order_by("sign1.fr.nnn", "sign1.fr.count").to_json(path)
    with open(path) as f:
        values = json.load(f)
    assert values == [
        {"sign1": {"label": f"label_{n}", "fr": {"nnn": f.nnn, "count": f.count}}}
        for n, f in enumerate(features)
    ]


# These deprecation warnings occur in the datamodel-code-generator package.
@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_to_from_jsonl(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    dc_to = DataChain.from_pandas(df, session=test_session)
    path = tmp_dir / "test.jsonl"
    dc_to.order_by("first_name", "age").to_jsonl(path)

    with open(path) as f:
        values = [json.loads(line) for line in f.read().split("\n")]
    assert values == [
        {"first_name": n, "age": a, "city": c}
        for n, a, c in zip(DF_DATA["first_name"], DF_DATA["age"], DF_DATA["city"])
    ]

    dc_from = DataChain.from_json(path.as_uri(), format="jsonl", session=test_session)
    df1 = dc_from.select("jsonl.first_name", "jsonl.age", "jsonl.city").to_pandas()
    df1 = df1["jsonl"]
    assert df_equal(df1, df)


# These deprecation warnings occur in the datamodel-code-generator package.
@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_from_jsonl_jmespath(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    values = [
        {"first_name": n, "age": a, "city": c}
        for n, a, c in zip(DF_DATA["first_name"], DF_DATA["age"], DF_DATA["city"])
    ]
    path = tmp_dir / "test.jsonl"
    with open(path, "w") as f:
        for v in values:
            f.write(
                json.dumps({"data": "Contained Within", "row_version": 5, "value": v})
            )
            f.write("\n")

    dc_from = DataChain.from_json(
        path.as_uri(), format="jsonl", jmespath="value", session=test_session
    )
    df1 = dc_from.select("value.first_name", "value.age", "value.city").to_pandas()
    df1 = df1["value"]
    assert df_equal(df1, df)


def test_to_jsonl_features(tmp_dir, test_session):
    dc_to = DataChain.from_values(
        f1=features, num=range(len(features)), session=test_session
    )
    path = tmp_dir / "test.json"
    dc_to.order_by("f1.nnn", "f1.count").to_jsonl(path)
    with open(path) as f:
        values = [json.loads(line) for line in f.read().split("\n")]
    assert values == [
        {"f1": {"nnn": f.nnn, "count": f.count}, "num": n}
        for n, f in enumerate(features)
    ]


def test_to_jsonl_features_nested(tmp_dir, test_session):
    dc_to = DataChain.from_values(sign1=features_nested, session=test_session)
    path = tmp_dir / "test.json"
    dc_to.order_by("sign1.fr.nnn", "sign1.fr.count").to_jsonl(path)
    with open(path) as f:
        values = [json.loads(line) for line in f.read().split("\n")]
    assert values == [
        {"sign1": {"label": f"label_{n}", "fr": {"nnn": f.nnn, "count": f.count}}}
        for n, f in enumerate(features)
    ]


def test_from_parquet(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    dc = DataChain.from_parquet(path.as_uri(), session=test_session)
    df1 = dc.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_from_parquet_in_memory(tmp_dir):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    dc = DataChain.from_parquet(path.as_uri(), in_memory=True)
    df1 = dc.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df)


def test_from_parquet_partitioned(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path, partition_cols=["first_name"])
    dc = DataChain.from_parquet(path.as_uri(), session=test_session)
    df1 = dc.select("first_name", "age", "city").to_pandas()
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    assert df_equal(df1, df)


def test_to_parquet(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    dc = DataChain.from_pandas(df, session=test_session)

    path = tmp_dir / "test.parquet"
    dc.to_parquet(path)

    assert path.is_file()
    pd.testing.assert_frame_equal(sort_df(pd.read_parquet(path)), sort_df(df))


def test_to_parquet_partitioned(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    dc = DataChain.from_pandas(df, session=test_session)

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


@pytest.mark.parametrize("chunk_size", (1000, 2))
@pytest.mark.parametrize("kwargs", ({}, {"compression": "gzip"}))
def test_to_from_parquet(tmp_dir, test_session, chunk_size, kwargs):
    df = pd.DataFrame(DF_DATA)
    dc_to = DataChain.from_pandas(df, session=test_session)

    path = tmp_dir / "test.parquet"
    dc_to.to_parquet(path, chunk_size=chunk_size, **kwargs)

    assert path.is_file()
    pd.testing.assert_frame_equal(sort_df(pd.read_parquet(path)), sort_df(df))

    dc_from = DataChain.from_parquet(path.as_uri(), session=test_session)
    df1 = dc_from.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df)


@pytest.mark.parametrize("chunk_size", (1000, 2))
def test_to_from_parquet_partitioned(tmp_dir, test_session, chunk_size):
    df = pd.DataFrame(DF_DATA)
    dc_to = DataChain.from_pandas(df, session=test_session)

    path = tmp_dir / "parquets"
    dc_to.to_parquet(path, partition_cols=["first_name"], chunk_size=chunk_size)

    assert set(path.iterdir()) == {
        path / f"first_name={name}" for name in df["first_name"]
    }
    df1 = pd.read_parquet(path)
    df1 = df1.reindex(columns=df.columns)
    df1["first_name"] = df1["first_name"].astype("str")
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df)

    dc_from = DataChain.from_parquet(path.as_uri(), session=test_session)
    df1 = dc_from.select("first_name", "age", "city").to_pandas()
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    assert df1.equals(df)


@pytest.mark.parametrize("chunk_size", (1000, 2))
def test_to_from_parquet_features(tmp_dir, test_session, chunk_size):
    dc_to = DataChain.from_values(
        f1=features, num=range(len(features)), session=test_session
    )

    path = tmp_dir / "test.parquet"
    dc_to.to_parquet(path, chunk_size=chunk_size)

    assert path.is_file()

    dc_from = DataChain.from_parquet(path.as_uri(), session=test_session)

    n = 0
    for sample in dc_from.order_by("f1.nnn", "f1.count").select("f1", "num").collect():
        assert len(sample) == 2
        fr, num = sample

        assert isinstance(fr, MyFr)
        assert isinstance(num, int)
        assert num == n
        assert fr == features[n]

        n += 1

    assert n == len(features)


@pytest.mark.parametrize("chunk_size", (1000, 2))
def test_to_from_parquet_nested_features(tmp_dir, test_session, chunk_size):
    dc_to = DataChain.from_values(sign1=features_nested, session=test_session)

    path = tmp_dir / "test.parquet"
    dc_to.to_parquet(path, chunk_size=chunk_size)

    assert path.is_file()

    dc_from = DataChain.from_parquet(path.as_uri(), session=test_session)

    for n, sample in enumerate(
        dc_from.order_by("sign1.fr.nnn", "sign1.fr.count").select("sign1").collect()
    ):
        assert len(sample) == 1
        nested = sample[0]

        assert isinstance(nested, MyNested)
        assert nested == features_nested[n]


@pytest.mark.parametrize("chunk_size", (1000, 2))
def test_to_from_parquet_two_top_level_features(tmp_dir, test_session, chunk_size):
    dc_to = DataChain.from_values(
        f1=features, nest1=features_nested, session=test_session
    )

    path = tmp_dir / "test.parquet"
    dc_to.to_parquet(path, chunk_size=chunk_size)

    assert path.is_file()

    dc_from = DataChain.from_parquet(path.as_uri(), session=test_session)

    for n, sample in enumerate(
        dc_from.order_by("f1.nnn", "f1.count").select("f1", "nest1").collect()
    ):
        assert len(sample) == 2
        fr, nested = sample

        assert isinstance(fr, MyFr)
        assert fr == features[n]
        assert isinstance(nested, MyNested)
        assert nested == features_nested[n]


@skip_if_not_sqlite
def test_parallel_in_memory():
    prefix = "t & "
    vals = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    with pytest.raises(RuntimeError):
        list(
            DataChain.from_values(key=vals, in_memory=True)
            .settings(parallel=True)
            .map(res=lambda key: prefix + key)
            .collect("res")
        )


def test_exec(test_session):
    names = ("f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json")
    all_names = set()

    dc = (
        DataChain.from_values(name=names, session=test_session)
        .map(nop=lambda name: all_names.add(name))
        .exec()
    )
    assert isinstance(dc, DataChain)
    assert all_names == set(names)


def test_extend_features(test_session):
    dc = DataChain.from_values(
        f1=features, num=range(len(features)), session=test_session
    )

    res = dc._extend_to_data_model("sum", "num")
    assert res == sum(range(len(features)))


def test_from_storage_object_name(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    dc = DataChain.from_storage(
        path.as_uri(), object_name="custom", session=test_session
    )
    assert dc.schema["custom"] == File


def test_from_features_object_name(test_session):
    fib = [1, 1, 2, 3, 5, 8]
    values = ["odd" if num % 2 else "even" for num in fib]

    dc = DataChain.from_values(
        fib=fib, odds=values, object_name="custom", session=test_session
    )
    assert "custom.fib" in dc.to_pandas(flatten=True).columns


def test_parse_tabular_object_name(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    dc = DataChain.from_storage(path.as_uri(), session=test_session).parse_tabular(
        object_name="tbl"
    )
    assert "tbl.first_name" in dc.to_pandas(flatten=True).columns


def test_sys_feature(test_session):
    ds = DataChain.from_values(t1=features, session=test_session).order_by(
        "t1.nnn", "t1.count"
    )
    ds_sys = ds.settings(sys=True)
    assert not ds._sys
    assert ds_sys._sys

    args = []
    ds_sys.map(res=lambda sys, t1: args.append((sys, t1))).save("ds_sys")

    sys_cls = Sys.model_construct
    assert args == [
        (sys_cls(id=1, rand=ANY), MyFr(nnn="n1", count=1)),
        (sys_cls(id=2, rand=ANY), MyFr(nnn="n1", count=3)),
        (sys_cls(id=3, rand=ANY), MyFr(nnn="n2", count=5)),
    ]
    assert "sys" not in test_session.catalog.get_dataset("ds_sys").feature_schema

    ds_no_sys = ds_sys.settings(sys=False)
    assert not ds_no_sys._sys

    args = []
    ds_no_sys.map(res=lambda t1: args.append(t1)).save("ds_no_sys")
    assert args == [
        MyFr(nnn="n1", count=1),
        MyFr(nnn="n1", count=3),
        MyFr(nnn="n2", count=5),
    ]
    assert "sys" not in test_session.catalog.get_dataset("ds_no_sys").feature_schema


def test_to_pandas_multi_level(test_session):
    df = DataChain.from_values(t1=features, session=test_session).to_pandas()

    assert "t1" in df.columns
    assert "nnn" in df["t1"].columns
    assert "count" in df["t1"].columns
    assert sort_df(df)["t1"]["count"].tolist() == [1, 3, 5]


def test_to_pandas_multi_level_flatten(test_session):
    df = DataChain.from_values(t1=features, session=test_session).to_pandas(
        flatten=True
    )

    assert "t1.nnn" in df.columns
    assert "t1.count" in df.columns
    assert len(df.columns) == 2
    assert sort_df(df)["t1.count"].tolist() == [1, 3, 5]


def test_to_pandas_empty(test_session):
    df = (
        DataChain.from_values(t1=[1, 2, 3], session=test_session)
        .limit(0)
        .to_pandas(flatten=True)
    )

    assert df.empty
    assert "t1" in df.columns
    assert df["t1"].tolist() == []

    df = (
        DataChain.from_values(my_n=features_nested, session=test_session)
        .limit(0)
        .to_pandas(flatten=False)
    )

    assert df.empty
    assert df["my_n"].empty
    assert list(df.columns) == [
        ("my_n", "label", ""),
        ("my_n", "fr", "nnn"),
        ("my_n", "fr", "count"),
    ]

    df = (
        DataChain.from_values(my_n=features_nested, session=test_session)
        .limit(0)
        .to_pandas(flatten=True)
    )

    assert df.empty
    assert df["my_n.fr.nnn"].tolist() == []
    assert list(df.columns) == ["my_n.label", "my_n.fr.nnn", "my_n.fr.count"]


def test_mutate(test_session):
    chain = (
        DataChain.from_values(t1=features, session=test_session)
        .order_by("t1.nnn", "t1.count")
        .mutate(circle=2 * 3.14 * Column("t1.count"), place="pref_" + Column("t1.nnn"))
    )

    assert chain.signals_schema.values["circle"] is float
    assert chain.signals_schema.values["place"] is str

    expected = [fr.count * 2 * 3.14 for fr in features]
    np.testing.assert_allclose(list(chain.collect("circle")), expected)


@pytest.mark.parametrize("with_function", [True, False])
def test_order_by_with_nested_columns(test_session, with_function):
    names = ["a.txt", "c.txt", "d.txt", "a.txt", "b.txt"]

    dc = DataChain.from_values(
        file=[File(path=name) for name in names], session=test_session
    )
    if with_function:
        from datachain.sql.functions.random import rand

        dc = dc.order_by("file.path", rand())
    else:
        dc = dc.order_by("file.path")

    assert list(dc.collect("file.path")) == [
        "a.txt",
        "a.txt",
        "b.txt",
        "c.txt",
        "d.txt",
    ]


def test_order_by_collect(test_session):
    numbers = [6, 2, 3, 1, 5, 7, 4]
    letters = ["u", "y", "x", "z", "v", "t", "w"]

    dc = DataChain.from_values(number=numbers, letter=letters, session=test_session)
    assert list(dc.order_by("number").collect()) == [
        (1, "z"),
        (2, "y"),
        (3, "x"),
        (4, "w"),
        (5, "v"),
        (6, "u"),
        (7, "t"),
    ]

    assert list(dc.order_by("letter").collect()) == [
        (7, "t"),
        (6, "u"),
        (5, "v"),
        (4, "w"),
        (3, "x"),
        (2, "y"),
        (1, "z"),
    ]


@pytest.mark.parametrize("with_function", [True, False])
def test_order_by_descending(test_session, with_function):
    names = ["a.txt", "c.txt", "d.txt", "a.txt", "b.txt"]

    dc = DataChain.from_values(
        file=[File(path=name) for name in names], session=test_session
    )
    if with_function:
        from datachain.sql.functions.random import rand

        dc = dc.order_by("file.path", rand(), descending=True)
    else:
        dc = dc.order_by("file.path", descending=True)

    assert list(dc.collect("file.path")) == [
        "d.txt",
        "c.txt",
        "b.txt",
        "a.txt",
        "a.txt",
    ]


def test_union(test_session):
    chain1 = DataChain.from_values(value=[1, 2], session=test_session)
    chain2 = DataChain.from_values(value=[3, 4], session=test_session)
    chain3 = chain1 | chain2
    assert chain3.count() == 4
    assert list(chain3.order_by("value").collect("value")) == [1, 2, 3, 4]


def test_union_different_columns(test_session):
    chain1 = DataChain.from_values(
        value=[1, 2], name=["chain", "more"], session=test_session
    )
    chain2 = DataChain.from_values(value=[3, 4], session=test_session)
    chain3 = DataChain.from_values(
        other=["a", "different", "thing"], session=test_session
    )
    with pytest.raises(
        ValueError, match="Cannot perform union. name only present in left"
    ):
        chain1.union(chain2).show()
    with pytest.raises(
        ValueError, match="Cannot perform union. name only present in right"
    ):
        chain2.union(chain1).show()
    with pytest.raises(
        ValueError,
        match="Cannot perform union. "
        "other only present in left. "
        "name, value only present in right",
    ):
        chain3.union(chain1).show()


def test_union_different_column_order(test_session):
    chain1 = DataChain.from_values(
        value=[1, 2], name=["chain", "more"], session=test_session
    )
    chain2 = DataChain.from_values(
        name=["different", "order"], value=[9, 10], session=test_session
    )
    assert list(chain1.union(chain2).order_by("value").collect()) == [
        (1, "chain"),
        (2, "more"),
        (9, "different"),
        (10, "order"),
    ]


def test_subtract(test_session):
    chain1 = DataChain.from_values(a=[1, 1, 2], b=["x", "y", "z"], session=test_session)
    chain2 = DataChain.from_values(a=[1, 2], b=["x", "y"], session=test_session)
    assert set(chain1.subtract(chain2, on=["a", "b"]).collect()) == {(1, "y"), (2, "z")}
    assert set(chain1.subtract(chain2, on=["b"]).collect()) == {(2, "z")}
    assert set(chain1.subtract(chain2, on=["a"]).collect()) == set()
    assert set(chain1.subtract(chain2).collect()) == {(1, "y"), (2, "z")}
    assert chain1.subtract(chain1).count() == 0

    chain3 = DataChain.from_values(a=[1, 3], c=["foo", "bar"], session=test_session)
    assert set(chain1.subtract(chain3, on="a").collect()) == {(2, "z")}
    assert set(chain1.subtract(chain3).collect()) == {(2, "z")}

    chain4 = DataChain.from_values(d=[1, 2, 3], e=["x", "y", "z"], session=test_session)
    chain5 = DataChain.from_values(a=[1, 2], b=["x", "y"], session=test_session)

    assert set(chain4.subtract(chain5, on="d", right_on="a").collect()) == {(3, "z")}


def test_subtract_error(test_session):
    chain1 = DataChain.from_values(a=[1, 1, 2], b=["x", "y", "z"], session=test_session)
    chain2 = DataChain.from_values(a=[1, 2], b=["x", "y"], session=test_session)
    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=[])
    with pytest.raises(TypeError):
        chain1.subtract(chain2, on=42)

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on="")

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on="a", right_on="")

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=["a", "b"], right_on=["c", ""])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=["a", "b"], right_on=[])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=["a", "b"], right_on=["d"])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on=[])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on="")

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on=42)

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on=["a"])

    with pytest.raises(TypeError):
        chain1.subtract(chain2, on=42, right_on=42)

    chain3 = DataChain.from_values(c=["foo", "bar"], session=test_session)
    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain3)


def test_column_math(test_session):
    fib = [1, 1, 2, 3, 5, 8]
    chain = DataChain.from_values(num=fib, session=test_session).order_by("num")

    ch = chain.mutate(add2=chain.column("num") + 2)
    assert list(ch.collect("add2")) == [x + 2 for x in fib]

    ch2 = ch.mutate(x=1 - ch.column("add2"))
    assert list(ch2.collect("x")) == [1 - (x + 2.0) for x in fib]


@skip_if_not_sqlite
def test_column_math_division(test_session):
    fib = [1, 1, 2, 3, 5, 8]
    chain = DataChain.from_values(num=fib, session=test_session)

    ch = chain.mutate(div2=chain.column("num") / 2.0)
    assert list(ch.collect("div2")) == [x / 2.0 for x in fib]


def test_from_values_array_of_floats(test_session):
    embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    chain = DataChain.from_values(emd=embeddings, session=test_session)

    assert list(chain.order_by("emd").collect("emd")) == embeddings


def test_custom_model_with_nested_lists(test_session):
    class Trace(BaseModel):
        x: float
        y: float

    class Nested(BaseModel):
        values: list[list[float]]
        traces_single: list[Trace]
        traces_double: list[list[Trace]]

    DataModel.register(Nested)

    ds = DataChain.from_values(
        nested=[
            Nested(
                values=[[0.5, 0.5], [0.5, 0.5]],
                traces_single=[{"x": 0.5, "y": 0.5}, {"x": 0.5, "y": 0.5}],
                traces_double=[[{"x": 0.5, "y": 0.5}], [{"x": 0.5, "y": 0.5}]],
            )
        ],
        nums=[1],
        session=test_session,
    )

    assert list(ds.collect("nested")) == [
        Nested(
            values=[[0.5, 0.5], [0.5, 0.5]],
            traces_single=[{"x": 0.5, "y": 0.5}, {"x": 0.5, "y": 0.5}],
            traces_double=[[{"x": 0.5, "y": 0.5}], [{"x": 0.5, "y": 0.5}]],
        )
    ]


def test_min_limit(test_session):
    dc = DataChain.from_values(a=[1, 2, 3, 4, 5], session=test_session)
    assert dc.count() == 5
    assert dc.limit(4).count() == 4
    assert dc.count() == 5
    assert dc.limit(1).count() == 1
    assert dc.count() == 5
    assert dc.limit(2).limit(3).count() == 2
    assert dc.count() == 5
    assert dc.limit(3).limit(2).count() == 2
    assert dc.count() == 5


def test_show_limit(test_session):
    dc = DataChain.from_values(a=[1, 2, 3, 4, 5], session=test_session)
    assert dc.count() == 5
    assert dc.limit(4).count() == 4
    dc.show(1)
    assert dc.count() == 5
    assert dc.limit(1).count() == 1
    dc.show(1)
    assert dc.count() == 5
    assert dc.limit(2).limit(3).count() == 2
    dc.show(1)
    assert dc.count() == 5
    assert dc.limit(3).limit(2).count() == 2
    dc.show(1)
    assert dc.count() == 5


def test_gen_limit(test_session):
    def func(key, val) -> Iterator[tuple[File, str]]:
        for i in range(val):
            yield File(path=""), f"{key}_{i}"

    keys = ["a", "b", "c", "d"]
    values = [3, 3, 3, 3]

    ds = DataChain.from_values(key=keys, val=values, session=test_session)

    assert ds.count() == 4
    assert ds.gen(res=func).count() == 12
    assert ds.limit(2).gen(res=func).count() == 6
    assert ds.limit(2).gen(res=func).limit(1).count() == 1
    assert ds.limit(3).gen(res=func).limit(2).count() == 2
    assert ds.limit(2).gen(res=func).limit(3).count() == 3
    assert ds.limit(3).gen(res=func).limit(10).count() == 9


def test_gen_filter(test_session):
    def func(key, val) -> Iterator[tuple[File, str]]:
        for i in range(val):
            yield File(path=""), f"{key}_{i}"

    keys = ["a", "b", "c", "d"]
    values = [3, 3, 3, 3]

    ds = DataChain.from_values(key=keys, val=values, session=test_session)

    assert ds.count() == 4
    assert ds.gen(res=func).count() == 12
    assert ds.gen(res=func).filter(C("res_1").glob("a_*")).count() == 3


def test_rename_non_object_column_name_with_mutate(test_session):
    ds = DataChain.from_values(ids=[1, 2, 3], session=test_session)
    ds = ds.mutate(my_ids=Column("ids"))

    assert ds.signals_schema.values == {"my_ids": int}
    assert list(ds.order_by("my_ids").collect("my_ids")) == [1, 2, 3]

    assert ds.signals_schema.values.get("my_ids") is int
    assert "ids" not in ds.signals_schema.values
    assert list(ds.order_by("my_ids").collect("my_ids")) == [1, 2, 3]


def test_rename_object_column_name_with_mutate(test_session):
    names = ["a", "b", "c"]
    sizes = [1, 2, 3]
    files = [File(path=name, size=size) for name, size in zip(names, sizes)]

    ds = DataChain.from_values(file=files, ids=[1, 2, 3], session=test_session)
    ds = ds.mutate(fname=Column("file.path"))

    assert list(ds.order_by("fname").collect("fname")) == ["a", "b", "c"]
    assert ds.signals_schema.values == {"file": File, "ids": int, "fname": str}

    # check that persist after saving
    ds.save("mutated")

    ds = DataChain.from_dataset(name="mutated", session=test_session)
    assert ds.signals_schema.values.get("file") is File
    assert ds.signals_schema.values.get("ids") is int
    assert ds.signals_schema.values.get("fname") is str
    assert list(ds.order_by("fname").collect("fname")) == ["a", "b", "c"]


def test_rename_object_name_with_mutate(test_session):
    names = ["a", "b", "c"]
    sizes = [1, 2, 3]
    files = [File(path=name, size=size) for name, size in zip(names, sizes)]

    ds = DataChain.from_values(file=files, ids=[1, 2, 3], session=test_session)
    ds = ds.mutate(my_file=Column("file"))

    assert list(ds.order_by("my_file.path").collect("my_file.path")) == ["a", "b", "c"]
    assert ds.signals_schema.values == {"my_file": File, "ids": int}

    # check that persist after saving
    ds.save("mutated")

    ds = DataChain.from_dataset(name="mutated", session=test_session)
    assert ds.signals_schema.values.get("my_file") is File
    assert ds.signals_schema.values.get("ids") is int
    assert "file" not in ds.signals_schema.values
    assert list(ds.order_by("my_file.path").collect("my_file.path")) == ["a", "b", "c"]


def test_column(test_session):
    ds = DataChain.from_values(
        ints=[1, 2],
        floats=[0.5, 0.5],
        file=[File(path="a"), File(path="b")],
        session=test_session,
    )

    c = ds.column("ints")
    assert isinstance(c, Column)
    assert c.name == "ints"
    assert isinstance(c.type, Int64)

    c = ds.column("floats")
    assert isinstance(c, Column)
    assert c.name == "floats"
    assert isinstance(c.type, Float)

    c = ds.column("file.path")
    assert isinstance(c, Column)
    assert c.name == "file__path"
    assert isinstance(c.type, String)

    with pytest.raises(ValueError):
        c = ds.column("missing")


def test_mutate_with_subtraction(test_session):
    ds = DataChain.from_values(id=[1, 2], session=test_session)
    assert ds.mutate(new=ds.column("id") - 1).signals_schema.values["new"] is int


def test_mutate_with_addition(test_session):
    ds = DataChain.from_values(id=[1, 2], session=test_session)
    assert ds.mutate(new=ds.column("id") + 1).signals_schema.values["new"] is int


def test_mutate_with_division(test_session):
    ds = DataChain.from_values(id=[1, 2], session=test_session)
    assert ds.mutate(new=ds.column("id") / 10).signals_schema.values["new"] is float


def test_mutate_with_multiplication(test_session):
    ds = DataChain.from_values(id=[1, 2], session=test_session)
    assert ds.mutate(new=ds.column("id") * 10).signals_schema.values["new"] is int


def test_mutate_with_sql_func(test_session):
    from datachain import func

    ds = DataChain.from_values(id=[1, 2], session=test_session)
    assert ds.mutate(new=func.avg("id")).signals_schema.values["new"] is float


def test_mutate_with_complex_expression(test_session):
    from datachain import func

    ds = DataChain.from_values(id=[1, 2], name=["Jim", "Jon"], session=test_session)
    assert (
        ds.mutate(new=func.sum("id") * (5 - func.min("id"))).signals_schema.values[
            "new"
        ]
        is int
    )


@skip_if_not_sqlite
def test_mutate_with_saving(test_session):
    ds = DataChain.from_values(id=[1, 2], session=test_session)
    ds = ds.mutate(new=ds.column("id") / 2).save("mutated")

    ds = DataChain.from_dataset(name="mutated", session=test_session)
    assert ds.signals_schema.values["new"] is float
    assert list(ds.collect("new")) == [0.5, 1.0]


def test_mutate_with_expression_without_type(test_session):
    with pytest.raises(DataChainColumnError) as excinfo:
        DataChain.from_values(id=[1, 2], session=test_session).mutate(
            new=(Column("id") - 1)
        ).save()

    assert str(excinfo.value) == (
        "Error for column new: Cannot infer type with expression id - :id_1"
    )


def test_from_values_nan_inf(test_session):
    vals = [float("nan"), float("inf"), float("-inf")]
    dc = DataChain.from_values(vals=vals, session=test_session)
    res = list(dc.collect("vals"))
    assert len(res) == 3
    assert any(r for r in res if np.isnan(r))
    assert any(r for r in res if np.isposinf(r))
    assert any(r for r in res if np.isneginf(r))


def test_from_pandas_nan_inf(test_session):
    vals = [float("nan"), float("inf"), float("-inf")]
    df = pd.DataFrame({"vals": vals})
    dc = DataChain.from_pandas(df, session=test_session)
    res = list(dc.collect("vals"))
    assert len(res) == 3
    assert any(r for r in res if np.isnan(r))
    assert any(r for r in res if np.isposinf(r))
    assert any(r for r in res if np.isneginf(r))


def test_from_parquet_nan_inf(tmp_dir, test_session):
    vals = [float("nan"), float("inf"), float("-inf")]
    tbl = pa.table({"vals": vals})
    path = tmp_dir / "test.parquet"
    pq.write_table(tbl, path)
    dc = DataChain.from_parquet(path.as_uri(), session=test_session)

    res = list(dc.collect("vals"))
    assert len(res) == 3
    assert any(r for r in res if np.isnan(r))
    assert any(r for r in res if np.isposinf(r))
    assert any(r for r in res if np.isneginf(r))


def test_from_csv_nan_inf(tmp_dir, test_session):
    vals = [float("nan"), float("inf"), float("-inf")]
    df = pd.DataFrame({"vals": vals})
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    dc = DataChain.from_csv(path.as_uri(), session=test_session)

    res = list(dc.collect("vals"))
    assert len(res) == 3
    assert any(r for r in res if np.isnan(r))
    assert any(r for r in res if np.isposinf(r))
    assert any(r for r in res if np.isneginf(r))


def test_from_hf(test_session):
    ds = Dataset.from_dict(DF_DATA)
    df = DataChain.from_hf(ds, session=test_session).to_pandas()
    assert df_equal(df, pd.DataFrame(DF_DATA))


def test_from_hf_object_name(test_session):
    ds = Dataset.from_dict(DF_DATA)
    df = DataChain.from_hf(ds, session=test_session, object_name="obj").to_pandas()
    assert df_equal(df["obj"], pd.DataFrame(DF_DATA))


def test_from_hf_invalid(test_session):
    with pytest.raises(FileNotFoundError):
        DataChain.from_hf("invalid_dataset", session=test_session)


def test_group_by_int(test_session):
    from datachain import func

    ds = (
        DataChain.from_values(
            col1=["a", "a", "b", "b", "b", "c"],
            col2=[1, 2, 3, 4, 5, 6],
            session=test_session,
        )
        .order_by("col1", "col2")
        .group_by(
            cnt=func.count(),
            cnt_col=func.count("col2"),
            sum=func.sum("col2"),
            avg=func.avg("col2"),
            min=func.min("col2"),
            max=func.max("col2"),
            value=func.any_value("col2"),
            collect=func.collect("col2"),
            partition_by="col1",
        )
    )

    assert ds.signals_schema.serialize() == {
        "col1": "str",
        "cnt": "int",
        "cnt_col": "int",
        "sum": "int",
        "avg": "float",
        "min": "int",
        "max": "int",
        "value": "int",
        "collect": "list[int]",
    }
    assert sorted_dicts(ds.to_records(), "col1") == sorted_dicts(
        [
            {
                "col1": "a",
                "cnt": 2,
                "cnt_col": 2,
                "sum": 3,
                "avg": 1.5,
                "min": 1,
                "max": 2,
                "value": ANY_VALUE(1, 2),
                "collect": [1, 2],
            },
            {
                "col1": "b",
                "cnt": 3,
                "cnt_col": 3,
                "sum": 12,
                "avg": 4.0,
                "min": 3,
                "max": 5,
                "value": ANY_VALUE(3, 4, 5),
                "collect": [3, 4, 5],
            },
            {
                "col1": "c",
                "cnt": 1,
                "cnt_col": 1,
                "sum": 6,
                "avg": 6.0,
                "min": 6,
                "max": 6,
                "value": ANY_VALUE(6),
                "collect": [6],
            },
        ],
        "col1",
    )


def test_group_by_float(test_session):
    from datachain import func

    ds = (
        DataChain.from_values(
            col1=["a", "a", "b", "b", "b", "c"],
            col2=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            session=test_session,
        )
        .order_by("col1", "col2")
        .group_by(
            cnt=func.count(),
            cnt_col=func.count("col2"),
            sum=func.sum("col2"),
            avg=func.avg("col2"),
            min=func.min("col2"),
            max=func.max("col2"),
            value=func.any_value("col2"),
            collect=func.collect("col2"),
            partition_by="col1",
        )
    )

    assert ds.signals_schema.serialize() == {
        "col1": "str",
        "cnt": "int",
        "cnt_col": "int",
        "sum": "float",
        "avg": "float",
        "min": "float",
        "max": "float",
        "value": "float",
        "collect": "list[float]",
    }
    assert sorted_dicts(ds.to_records(), "col1") == sorted_dicts(
        [
            {
                "col1": "a",
                "cnt": 2,
                "cnt_col": 2,
                "sum": 4.0,
                "avg": 2.0,
                "min": 1.5,
                "max": 2.5,
                "value": ANY_VALUE(1.5, 2.5),
                "collect": [1.5, 2.5],
            },
            {
                "col1": "b",
                "cnt": 3,
                "cnt_col": 3,
                "sum": 13.5,
                "avg": 4.5,
                "min": 3.5,
                "max": 5.5,
                "value": ANY_VALUE(3.5, 4.5, 5.5),
                "collect": [3.5, 4.5, 5.5],
            },
            {
                "col1": "c",
                "cnt": 1,
                "cnt_col": 1,
                "sum": 6.5,
                "avg": 6.5,
                "min": 6.5,
                "max": 6.5,
                "value": ANY_VALUE(6.5),
                "collect": [6.5],
            },
        ],
        "col1",
    )


def test_group_by_str(test_session):
    from datachain import func

    ds = (
        DataChain.from_values(
            col1=["a", "a", "b", "b", "b", "c"],
            col2=["1", "2", "3", "4", "5", "6"],
            session=test_session,
        )
        .order_by("col1", "col2")
        .group_by(
            cnt=func.count(),
            cnt_col=func.count("col2"),
            min=func.min("col2"),
            max=func.max("col2"),
            concat=func.concat("col2"),
            concat_sep=func.concat("col2", separator=","),
            value=func.any_value("col2"),
            collect=func.collect("col2"),
            partition_by="col1",
        )
    )

    assert ds.signals_schema.serialize() == {
        "col1": "str",
        "cnt": "int",
        "cnt_col": "int",
        "min": "str",
        "max": "str",
        "concat": "str",
        "concat_sep": "str",
        "value": "str",
        "collect": "list[str]",
    }
    assert sorted_dicts(ds.to_records(), "col1") == sorted_dicts(
        [
            {
                "col1": "a",
                "cnt": 2,
                "cnt_col": 2,
                "min": "1",
                "max": "2",
                "concat": "12",
                "concat_sep": "1,2",
                "value": ANY_VALUE("1", "2"),
                "collect": ["1", "2"],
            },
            {
                "col1": "b",
                "cnt": 3,
                "cnt_col": 3,
                "min": "3",
                "max": "5",
                "concat": "345",
                "concat_sep": "3,4,5",
                "value": ANY_VALUE("3", "4", "5"),
                "collect": ["3", "4", "5"],
            },
            {
                "col1": "c",
                "cnt": 1,
                "cnt_col": 1,
                "min": "6",
                "max": "6",
                "concat": "6",
                "concat_sep": "6",
                "value": ANY_VALUE("6"),
                "collect": ["6"],
            },
        ],
        "col1",
    )


def test_group_by_multiple_partition_by(test_session):
    from datachain import func

    ds = (
        DataChain.from_values(
            col1=["a", "a", "b", "b", "b", "c"],
            col2=[1, 2, 1, 2, 1, 2],
            col3=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            col4=["1", "2", "3", "4", "5", "6"],
            session=test_session,
        )
        .order_by("col1", "col2", "col3", "col4")
        .group_by(
            cnt=func.count(),
            cnt_col=func.count("col2"),
            sum=func.sum("col3"),
            concat=func.concat("col4"),
            value=func.any_value("col3"),
            collect=func.collect("col3"),
            partition_by=("col1", "col2"),
        )
    )

    assert ds.signals_schema.serialize() == {
        "col1": "str",
        "col2": "int",
        "cnt": "int",
        "cnt_col": "int",
        "sum": "float",
        "concat": "str",
        "value": "float",
        "collect": "list[float]",
    }
    assert sorted_dicts(ds.to_records(), "col1", "col2") == sorted_dicts(
        [
            {
                "col1": "a",
                "col2": 1,
                "cnt": 1,
                "cnt_col": 1,
                "sum": 1.0,
                "concat": "1",
                "value": ANY_VALUE(1.0),
                "collect": [1.0],
            },
            {
                "col1": "a",
                "col2": 2,
                "cnt": 1,
                "cnt_col": 1,
                "sum": 2.0,
                "concat": "2",
                "value": ANY_VALUE(2.0),
                "collect": [2.0],
            },
            {
                "col1": "b",
                "col2": 1,
                "cnt": 2,
                "cnt_col": 2,
                "sum": 8.0,
                "concat": "35",
                "value": ANY_VALUE(3.0, 5.0),
                "collect": [3.0, 5.0],
            },
            {
                "col1": "b",
                "col2": 2,
                "cnt": 1,
                "cnt_col": 1,
                "sum": 4.0,
                "concat": "4",
                "value": ANY_VALUE(4.0),
                "collect": [4.0],
            },
            {
                "col1": "c",
                "col2": 2,
                "cnt": 1,
                "cnt_col": 1,
                "sum": 6.0,
                "concat": "6",
                "value": ANY_VALUE(6.0),
                "collect": [6.0],
            },
        ],
        "col1",
        "col2",
    )


def test_group_by_no_partition_by(test_session):
    from datachain import func

    ds = (
        DataChain.from_values(
            col1=["a", "a", "b", "b", "b", "c"],
            col2=[1, 2, 1, 2, 1, 2],
            col3=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            col4=["1", "2", "3", "4", "5", "6"],
            session=test_session,
        )
        .order_by("col4")
        .group_by(
            cnt=func.count(),
            cnt_col=func.count("col2"),
            sum=func.sum("col3"),
            concat=func.concat("col4"),
            value=func.any_value("col3"),
            collect=func.collect("col3"),
        )
    )

    assert ds.signals_schema.serialize() == {
        "cnt": "int",
        "cnt_col": "int",
        "sum": "float",
        "concat": "str",
        "value": "float",
        "collect": "list[float]",
    }
    assert ds.to_records() == [
        {
            "cnt": 6,
            "cnt_col": 6,
            "sum": 21.0,
            "concat": "123456",
            "value": 1.0,
            "collect": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
    ]


def test_group_by_schema(test_session):
    from datachain import func

    class Signal(DataModel):
        name: str
        value: float

    class Parent(DataModel):
        signal: Signal

    def multiplier(name: str, val: float) -> Signal:
        return Signal(name=name, value=val * 2)

    def to_parent(signal: Signal) -> Parent:
        return Parent(signal=signal)

    dc = (
        DataChain.from_values(
            name=["a", "a", "b", "b", "b", "c"],
            val=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .map(signal=multiplier)
        .map(parent=to_parent)
        .order_by("signal.name")
        .group_by(
            cnt=func.count(),
            sum=func.sum("signal.value"),
            partition_by=["signal.name", "parent.signal.name"],
        )
        .select("signal.name", "parent.signal.name", "cnt", "sum")
    )

    assert dc.signals_schema.serialize() == {
        "signal.name": "str",
        "parent.signal.name": "str",
        "cnt": "int",
        "sum": "float",
    }
    assert dc.to_records() == [
        {
            "signal__name": "a",
            "parent__signal__name": "a",
            "cnt": 2,
            "sum": 6.0,
        },
        {
            "signal__name": "b",
            "parent__signal__name": "b",
            "cnt": 3,
            "sum": 24.0,
        },
        {
            "signal__name": "c",
            "parent__signal__name": "c",
            "cnt": 1,
            "sum": 12.0,
        },
    ]


def test_group_by_error(test_session):
    from datachain import func

    dc = DataChain.from_values(
        col1=["a", "a", "b", "b", "b", "c"],
        col2=[1, 2, 3, 4, 5, 6],
        session=test_session,
    )

    with pytest.raises(
        ValueError, match="At least one column should be provided for group_by"
    ):
        dc.group_by(partition_by="col1")

    with pytest.raises(
        DataChainColumnError,
        match="Column foo has type <class 'str'> but expected Func object",
    ):
        dc.group_by(foo="col2", partition_by="col1")

    with pytest.raises(
        SignalResolvingError, match="cannot resolve signal name 'col3': is not found"
    ):
        dc.group_by(foo=func.sum("col3"), partition_by="col1")

    with pytest.raises(
        SignalResolvingError, match="cannot resolve signal name 'col3': is not found"
    ):
        dc.group_by(foo=func.sum("col2"), partition_by="col3")


def test_group_by_case(test_session):
    from datachain import func

    ds = DataChain.from_values(
        col1=[1.0, 0.0, 3.2, 0.1, 5.9, -1.0],
        col2=[0.0, 6.1, -0.05, 3.7, 0.1, -3.0],
        session=test_session,
    ).group_by(
        col1=func.sum(func.case((C("col1") > 0.1, 1), else_=0)),
        col2=func.sum(func.case((C("col2") < 0.0, 1), else_=0)),
    )

    assert ds.signals_schema.serialize() == {
        "col1": "int",
        "col2": "int",
    }
    assert ds.to_records() == [
        {
            "col1": 3,
            "col2": 2,
        }
    ]


@pytest.mark.parametrize("desc", [True, False])
def test_window_functions(test_session, desc):
    from datachain import func

    window = func.window(partition_by="col1", order_by="col2", desc=desc)

    ds = DataChain.from_values(
        col1=["a", "a", "b", "b", "b", "c"],
        col2=[1, 2, 3, 4, 5, 6],
        session=test_session,
    ).mutate(
        row_number=func.row_number().over(window),
        rank=func.rank().over(window),
        dense_rank=func.dense_rank().over(window),
        first=func.first("col2").over(window),
    )

    assert ds.signals_schema.serialize() == {
        "col1": "str",
        "col2": "int",
        "row_number": "int",
        "rank": "int",
        "dense_rank": "int",
        "first": "int",
    }
    assert sorted_dicts(ds.to_records(), "col1", "col2") == sorted_dicts(
        [
            {
                "col1": "a",
                "col2": 1,
                "row_number": 2 if desc else 1,
                "rank": 2 if desc else 1,
                "dense_rank": 2 if desc else 1,
                "first": 2 if desc else 1,
            },
            {
                "col1": "a",
                "col2": 2,
                "row_number": 1 if desc else 2,
                "rank": 1 if desc else 2,
                "dense_rank": 1 if desc else 2,
                "first": 2 if desc else 1,
            },
            {
                "col1": "b",
                "col2": 3,
                "row_number": 3 if desc else 1,
                "rank": 3 if desc else 1,
                "dense_rank": 3 if desc else 1,
                "first": 5 if desc else 3,
            },
            {
                "col1": "b",
                "col2": 4,
                "row_number": 2,
                "rank": 2,
                "dense_rank": 2,
                "first": 5 if desc else 3,
            },
            {
                "col1": "b",
                "col2": 5,
                "row_number": 1 if desc else 3,
                "rank": 1 if desc else 3,
                "dense_rank": 1 if desc else 3,
                "first": 5 if desc else 3,
            },
            {
                "col1": "c",
                "col2": 6,
                "row_number": 1,
                "rank": 1,
                "dense_rank": 1,
                "first": 6,
            },
        ],
        "col1",
        "col2",
    )


def test_window_error(test_session):
    from datachain import func

    window = func.window(partition_by="col1", order_by="col2")

    dc = DataChain.from_values(
        col1=["a", "a", "b", "b", "b", "c"],
        col2=[1, 2, 3, 4, 5, 6],
        session=test_session,
    )

    with pytest.raises(
        DataChainParamsError,
        match=re.escape(
            "Window function first() requires over() clause with a window spec",
        ),
    ):
        dc.mutate(first=func.first("col2"))

    with pytest.raises(
        DataChainParamsError,
        match=re.escape(
            "sum() doesn't support window (over())",
        ),
    ):
        dc.mutate(first=func.sum("col2").over(window))
