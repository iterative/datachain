import functools
import os
import pickle
import posixpath

import pytest

import datachain as dc
from datachain.func import path as pathfunc
from datachain.lib.file import AudioFile, AudioFragment, File
from datachain.lib.udf import Mapper
from datachain.lib.utils import DataChainError
from tests.utils import LARGE_TREE, NUM_TREE


def test_udf_none_nested_datamodel_after_outer_merge(test_session):
    """
    Test that UDFs can handle None values for nested DataModel objects
    """
    # Get warehouse default values for proper NULL handling checks
    catalog = test_session.catalog
    from datachain.sql.types import Int, String

    int_default = Int.default_value(catalog.warehouse.db.dialect)
    str_default = String.default_value(catalog.warehouse.db.dialect)

    # Create sample data with AudioFragment (which has nested AudioFile)
    left = dc.read_values(
        id=[1, 2],
        audio_fragment=[
            AudioFragment(
                audio=AudioFile(path="audio1.wav", source="file://"),
                start=0.0,
                end=1.0,
            ),
            AudioFragment(
                audio=AudioFile(path="audio2.wav", source="file://"),
                start=0.0,
                end=1.0,
            ),
        ],
        session=test_session,
    )

    right = dc.read_values(
        id=[2, 3],
        audio_info=[
            AudioFragment(
                audio=AudioFile(path="audio2_right.wav", source="file://"),
                start=1.0,
                end=2.0,
            ),
            AudioFragment(
                audio=AudioFile(path="audio3_right.wav", source="file://"),
                start=0.0,
                end=1.0,
            ),
        ],
        session=test_session,
    )

    # Full outer merge creates None (or default values in CH) for unmatched sides
    merged = left.merge(right, on="id", full=True)
    assert merged.count() == 3

    def extract_paths(
        id: int, right_id: int, audio_fragment: AudioFragment, audio_info: AudioFragment
    ) -> tuple:
        left_path = audio_fragment.audio.path if audio_fragment else None
        right_path = audio_info.audio.path if audio_info else None
        extracted_file = audio_fragment.audio if audio_fragment else None
        return id, right_id, left_path, right_path, extracted_file

    result = merged.settings(prefetch=False).map(
        extract_paths,
        params=["id", "right_id", "audio_fragment", "audio_info"],
        output={
            "merged_id": int,
            "merged_right_id": int,
            "left_path": str,
            "right_path": str,
            "extracted_file": AudioFile,
        },
    )
    rows = sorted(
        result.select(
            "merged_id",
            "merged_right_id",
            "left_path",
            "right_path",
            "extracted_file",
        ).to_iter(),
        key=lambda r: (
            r[0] if r[0] is not None else -1,
            r[1] if r[1] is not None else -1,
        ),
    )

    assert len(rows) == 3

    # Row with right-only data: left id=int_default (NULL), right_id=3
    assert rows[0][0] == int_default  # merged_id (NULL → int_default)
    assert rows[0][1] == 3  # merged_right_id
    assert rows[0][2] == str_default  # left_path (NULL → str_default)
    assert rows[0][3] == "audio3_right.wav"  # right_path
    # extracted_file: NULL AudioFile → None on SQLite, default AudioFile on ClickHouse
    if rows[0][4] is None:
        # SQLite: NULL object becomes None
        assert rows[0][4] is None
    else:
        # ClickHouse: NULL object becomes object with default values
        assert isinstance(rows[0][4], AudioFile)
        assert rows[0][4].path == str_default

    # Row with left-only data: left id=1, right_id=0 (NULL)
    assert rows[1][0] == 1
    assert rows[1][1] == int_default
    assert rows[1][2] == "audio1.wav"
    assert rows[1][3] == str_default
    assert isinstance(rows[1][4], AudioFile)
    assert rows[1][4].path == "audio1.wav"
    assert rows[1][4].source == "file://"

    # Row with matched data: id=2, right_id=2 (both sides present)
    assert rows[2][0] == 2
    assert rows[2][1] == 2
    assert rows[2][2] == "audio2.wav"
    assert rows[2][3] == "audio2_right.wav"
    assert isinstance(rows[2][4], AudioFile)
    assert rows[2][4].path == "audio2.wav"
    assert rows[2][4].source == "file://"


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_udf(cloud_test_catalog):
    session = cloud_test_catalog.session

    def name_len(path):
        return (len(posixpath.basename(path)),)

    chain = (
        dc.read_storage(cloud_test_catalog.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .map(name_len, params=["file.path"], output={"name_len": int})
    )
    result1 = chain.select("file.path", "name_len").to_list()
    # ensure that we're able to run with same query multiple times
    result2 = chain.select("file.path", "name_len").to_list()
    count = chain.count()
    assert len(result1) == 3
    assert len(result2) == 3
    assert count == 3

    for r1, r2 in zip(result1, result2, strict=False):
        # Check that the UDF ran successfully
        assert len(posixpath.basename(r1[0])) == r1[1]
        assert len(posixpath.basename(r2[0])) == r2[1]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_parallel(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    def name_len(name):
        return (len(name),)

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .settings(parallel=True)
        .map(name_len, params=["file.path"], output={"name_len": int})
        .select("file.path", "name_len")
    )

    # Check that the UDF ran successfully
    count = 0
    for r in chain:
        count += 1
        assert len(r[0]) == r[1]
    assert count == 7


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_class_udf(cloud_test_catalog):
    session = cloud_test_catalog.session

    class MyUDF(Mapper):
        def __init__(self, constant, multiplier=1):
            self.constant = constant
            self.multiplier = multiplier

        def process(self, size):
            return (self.constant + size * self.multiplier,)

    chain = (
        dc.read_storage(cloud_test_catalog.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .map(
            MyUDF(5, multiplier=2),
            output={"total": int},
            params=["file.size"],
        )
        .select("file.size", "total")
        .order_by("file.size")
    )

    assert chain.to_list() == [
        (3, 11),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_class_udf_parallel(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    class MyUDF(Mapper):
        def __init__(self, constant, multiplier=1):
            self.constant = constant
            self.multiplier = multiplier

        def process(self, size):
            return (self.constant + size * self.multiplier,)

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .settings(parallel=2)
        .map(
            MyUDF(5, multiplier=2),
            output={"total": int},
            params=["file.size"],
        )
        .select("file.size", "total")
        .order_by("file.size")
    )

    assert chain.to_list() == [
        (3, 11),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, NUM_TREE), ("file", False, NUM_TREE)],
    indirect=True,
)
def test_udf_after_limit(cloud_test_catalog):
    ctc = cloud_test_catalog

    def name_int(name: str) -> int:
        try:
            return int(name)
        except ValueError:
            return 0

    def get_result(chain):
        res = chain.limit(100).map(name_int=name_int).order_by("name")
        return res.to_list("name", "name_int")

    expected = [(f"{i:06d}", i) for i in range(100)]
    chain = (
        dc.read_storage(ctc.src_uri, session=ctc.session)
        .mutate(name=pathfunc.name("file.path"))
        .persist()
    )
    # We test a few different orderings here, because we've had strange
    # bugs in the past where calling add_signals() after limit() gave us
    # incorrect results on clickhouse cloud.
    # See https://github.com/iterative/dvcx/issues/940
    assert get_result(chain.order_by("name")) == expected
    assert len(get_result(chain.order_by("sys.rand"))) == 100
    assert len(get_result(chain)) == 100


def test_udf_different_types(cloud_test_catalog):
    obj = {"name": "John", "age": 30}

    def test_types():
        return (
            5,
            5,
            5,
            0.5,
            0.5,
            0.5,
            [0.5],
            [[0.5], [0.5]],
            [0.5],
            [0.5],
            "s",
            True,
            {"a": 1},
            pickle.dumps(obj),
        )

    chain = (
        dc.read_storage(cloud_test_catalog.src_uri, session=cloud_test_catalog.session)
        .filter(dc.C("file.path").glob("*cat1"))
        .map(
            test_types,
            params=[],
            output={
                "int_col": int,
                "int_col_32": int,
                "int_col_64": int,
                "float_col": float,
                "float_col_32": float,
                "float_col_64": float,
                "array_col": list[float],
                "array_col_nested": list[list[float]],
                "array_col_32": list[float],
                "array_col_64": list[float],
                "string_col": str,
                "bool_col": bool,
                "json_col": dict,
                "binary_col": bytes,
            },
        )
    )

    results = chain.to_records()
    col_values = [
        (
            r["int_col"],
            r["int_col_32"],
            r["int_col_64"],
            r["float_col"],
            r["float_col_32"],
            r["float_col_64"],
            r["array_col"],
            r["array_col_nested"],
            r["array_col_32"],
            r["array_col_64"],
            r["string_col"],
            r["bool_col"],
            r["json_col"],
            pickle.loads(r["binary_col"]),  # noqa: S301
        )
        for r in results
    ]

    assert col_values == [
        (
            5,
            5,
            5,
            0.5,
            0.5,
            0.5,
            [0.5],
            [[0.5], [0.5]],
            [0.5],
            [0.5],
            "s",
            True,
            {"a": 1},
            obj,
        )
    ]


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("prefetch", [0, 2])
def test_map_file(cloud_test_catalog, use_cache, prefetch, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    ctc = cloud_test_catalog
    ctc.catalog.cache.clear()

    def is_prefetched(file: File) -> bool:
        return file._catalog.cache.contains(file) and bool(file.get_local_path())

    def verify_cache_used(file):
        catalog = file._catalog
        if use_cache or not prefetch:
            assert catalog.cache == cloud_test_catalog.catalog.cache
            return
        head, tail = os.path.split(catalog.cache.cache_dir)
        assert head == catalog.cache.tmp_dir
        assert tail.startswith("prefetch-")

    def with_checks(func, seen=[]):  # noqa: B006
        @functools.wraps(func)
        def wrapped(file, *args, **kwargs):
            # previously prefetched files should be removed if `cache` is disabled.
            for f in seen:
                assert f._catalog.cache.contains(f) == use_cache
            seen.append(file)

            assert is_prefetched(file) == (prefetch > 0)
            verify_cache_used(file)
            return func(file, *args, **kwargs)

        return wrapped

    def new_signal(file: File) -> str:
        with file.open() as f:
            return file.name + " -> " + f.read().decode("utf-8")

    chain = (
        dc.read_storage(ctc.src_uri, session=ctc.session)
        .settings(cache=use_cache, prefetch=prefetch)
        .map(signal=with_checks(new_signal))
        .persist()
    )

    expected = {
        "description -> Cats and Dogs",
        "cat1 -> meow",
        "cat2 -> mrow",
        "dog1 -> woof",
        "dog2 -> arf",
        "dog3 -> bark",
        "dog4 -> ruff",
    }
    assert set(chain.to_values("signal")) == expected
    for file in chain.to_values("file"):
        assert bool(file.get_local_path()) is use_cache
    assert not os.listdir(ctc.catalog.cache.tmp_dir)


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("prefetch", [0, 2])
def test_gen_file(cloud_test_catalog, use_cache, prefetch, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    ctc = cloud_test_catalog
    ctc.catalog.cache.clear()

    def is_prefetched(file: File) -> bool:
        return file._catalog.cache.contains(file) and bool(file.get_local_path())

    def verify_cache_used(file):
        catalog = file._catalog
        if use_cache or not prefetch:
            assert catalog.cache == cloud_test_catalog.catalog.cache
            return
        head, tail = os.path.split(catalog.cache.cache_dir)
        assert head == catalog.cache.tmp_dir
        assert tail.startswith("prefetch-")

    def with_checks(func, seen=[]):  # noqa: B006
        @functools.wraps(func)
        def wrapped(file, *args, **kwargs):
            # previously prefetched files should be removed if `cache` is disabled.
            for f in seen:
                assert f._catalog.cache.contains(f) == use_cache
            seen.append(file)

            assert is_prefetched(file) == (prefetch > 0)
            verify_cache_used(file)
            return func(file, *args, **kwargs)

        return wrapped

    def new_signal(file: File) -> list[str]:
        with file.open("rb") as f:
            return [file.name, f.read().decode("utf-8")]

    chain = (
        dc.read_storage(ctc.src_uri, session=ctc.session)
        .settings(cache=use_cache, prefetch=prefetch)
        .gen(signal=with_checks(new_signal), output=str)
        .persist()
    )
    expected = {
        "Cats and Dogs",
        "arf",
        "bark",
        "cat1",
        "cat2",
        "description",
        "dog1",
        "dog2",
        "dog3",
        "dog4",
        "meow",
        "mrow",
        "ruff",
        "woof",
    }
    assert set(chain.to_values("signal")) == expected
    assert not os.listdir(ctc.catalog.cache.tmp_dir)


def test_batch_for_map(test_session):
    # Create a chain with batch settings
    chain = dc.read_values(x=list(range(100)), session=test_session)
    chain_with_settings = chain.settings(batch_size=15)

    def add_one(x):
        return x + 1

    result = chain_with_settings.map(add_one, output={"result": int})

    results = [
        r[0] for r in result.to_iter("result")
    ]  # Access the first element of each tuple

    assert len(results) == 100
    assert set(results) == set(range(1, 101))


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_parallel_exec_error(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    def name_len_error(_name):
        # A udf that raises an exception
        raise RuntimeError("Test Error!")

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .settings(parallel=True)
        .map(name_len_error, params=["file.path"], output={"name_len": int})
    )

    if os.environ.get("DATACHAIN_DISTRIBUTED"):
        # in distributed mode we expect DataChainError with the error message
        with pytest.raises(DataChainError, match="Test Error!"):
            chain.show()
    else:
        # while in local mode we expect RuntimeError with the error message
        with pytest.raises(RuntimeError, match="UDF Execution Failed!"):
            chain.show()


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_reuse_on_error(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    error_state = {"error": True}

    def name_len_maybe_error(path):
        if error_state["error"]:
            # A udf that raises an exception
            raise RuntimeError("Test Error!")
        return (len(path),)

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .map(name_len_maybe_error, params=["file.path"], output={"path_len": int})
        .select("file.path", "path_len")
    )

    with pytest.raises(DataChainError, match="Test Error!"):
        chain.show()

    # Simulate fixing the error
    error_state["error"] = False

    # Retry Query
    count = 0
    for r in chain:
        # Check that the UDF ran successfully
        count += 1
        assert len(r[0]) == r[1]
    assert count == 3


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_parallel_interrupt(cloud_test_catalog_tmpfile, capfd):
    session = cloud_test_catalog_tmpfile.session

    def name_len_interrupt(_name):
        # A UDF that emulates cancellation due to a KeyboardInterrupt.
        raise KeyboardInterrupt

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .settings(parallel=True)
        .map(name_len_interrupt, params=["file.path"], output={"name_len": int})
    )
    if os.environ.get("DATACHAIN_DISTRIBUTED"):
        with pytest.raises(KeyboardInterrupt):
            chain.show()
    else:
        with pytest.raises(RuntimeError, match="UDF Execution Failed!"):
            chain.show()
    captured = capfd.readouterr()
    assert "semaphore" not in captured.err


@pytest.mark.xdist_group(name="tmpfile")
def test_udf_parallel_boostrap(test_session_tmpfile):
    vals = ["a", "b", "c", "d", "e", "f"]

    class MyMapper(Mapper):
        DEFAULT_VALUE = 84
        BOOTSTRAP_VALUE = 1452
        TEARDOWN_VALUE = 98763

        def __init__(self):
            super().__init__()
            self.value = MyMapper.DEFAULT_VALUE
            self._had_teardown = False

        def process(self, key) -> int:
            return self.value

        def setup(self):
            self.value = MyMapper.BOOTSTRAP_VALUE

        def teardown(self):
            self.value = MyMapper.TEARDOWN_VALUE

    chain = dc.read_values(key=vals, session=test_session_tmpfile)

    res = chain.settings(parallel=4).map(res=MyMapper()).to_values("res")

    assert res == [MyMapper.BOOTSTRAP_VALUE] * len(vals)


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, LARGE_TREE)],
    indirect=True,
)
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.parametrize("parallel", (1, 2))
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_distributed(
    cloud_test_catalog_tmpfile, workers, parallel, tree, run_datachain_worker
):
    session = cloud_test_catalog_tmpfile.session

    def name_len(name):
        return (len(name),)

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .settings(parallel=parallel, workers=workers)
        .map(name_len, params=["file.path"], output={"name_len": int})
        .select("file.path", "name_len")
    )

    # Check that the UDF ran successfully
    count = 0
    for r in chain:
        count += 1
        assert len(r[0]) == r[1]
    assert count == 225


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, LARGE_TREE)],
    indirect=True,
)
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.parametrize("parallel", (1, 2))
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_distributed_exec_error(
    cloud_test_catalog_tmpfile, workers, parallel, tree, run_datachain_worker
):
    session = cloud_test_catalog_tmpfile.session

    def name_len_error(_name):
        # A udf that raises an exception
        raise RuntimeError("Test Error!")

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .settings(parallel=parallel, workers=workers)
        .map(name_len_error, params=["file.path"], output={"name_len": int})
    )
    with pytest.raises(DataChainError, match="Test Error!"):
        chain.show()


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, LARGE_TREE)],
    indirect=True,
)
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.parametrize("parallel", (1, 2))
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_distributed_interrupt(
    cloud_test_catalog_tmpfile, capfd, tree, workers, parallel, run_datachain_worker
):
    session = cloud_test_catalog_tmpfile.session

    def name_len_interrupt(_name):
        # A UDF that emulates cancellation due to a KeyboardInterrupt.
        raise KeyboardInterrupt

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .settings(parallel=parallel, workers=workers)
        .map(name_len_interrupt, params=["file.path"], output={"name_len": int})
    )
    with pytest.raises(KeyboardInterrupt):
        chain.show()
    captured = capfd.readouterr()
    assert "semaphore" not in captured.err
