from collections.abc import Iterable
from typing import Union

import pytest
from pydantic import BaseModel

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.utils import normalize_col_names, rebase_path
from datachain.sql.types import Array, String


class MyModel(BaseModel):
    val1: str


class MyFeature(BaseModel):
    val1: str


@pytest.mark.parametrize(
    "typ,expected",
    (
        (list[str], Array(String())),
        (Iterable[str], Array(String())),
        (list[list[str]], Array(Array(String()))),
    ),
)
def test_convert_type_to_datachain_array(typ, expected):
    assert python_to_sql(typ).to_dict() == expected.to_dict()


@pytest.mark.parametrize(
    "typ",
    (
        Union[str, int],
        list[Union[str, int]],
        MyFeature,
        MyModel,
    ),
)
def test_convert_type_to_datachain_error(typ):
    with pytest.raises(TypeError):
        python_to_sql(typ)


def test_normalize_column_names():
    res = normalize_col_names(
        [
            "UpperCase",
            "_underscore_start",
            "double__underscore",
            "1start_with_number",
            "не_ascii_start",
            "  space_start",
            "space_end  ",
            "dash-end-",
            "-dash-start",
            "--multiple--dash--",
            "-_ mix_  -dash_ -",
            "__2digit_after_uderscore",
            "",
            "_-_-  _---_ _",
            "_-_-  _---_ _1",
        ]
    )
    assert list(res.keys()) == [
        "uppercase",
        "underscore_start",
        "double_underscore",
        "c0_1start_with_number",
        "ascii_start",
        "space_start",
        "space_end",
        "dash_end",
        "dash_start",
        "multiple_dash",
        "mix_dash",
        "c1_2digit_after_uderscore",
        "c2",
        "c3",
        "c4_1",
    ]


def test_normalize_column_names_case_repeat():
    res = normalize_col_names(["UpperCase", "UpPerCase"])

    assert list(res.keys()) == ["uppercase", "c0_uppercase"]


def test_normalize_column_names_exists_after_normalize():
    res = normalize_col_names(["1digit", "c0_1digit"])

    assert list(res.keys()) == ["c1_1digit", "c0_1digit"]


def test_normalize_column_names_normalized_repeat():
    res = normalize_col_names(["column", "_column"])

    assert list(res.keys()) == ["column", "c0_column"]


def test_normalize_column_names_normalized_case_repeat():
    res = normalize_col_names(["CoLuMn", "_column"])

    assert res == {"column": "CoLuMn", "c0_column": "_column"}


def test_normalize_column_names_repeat_generated_after_normalize():
    res = normalize_col_names(["c0_CoLuMn", "_column", "column"])

    assert res == {"c0_column": "c0_CoLuMn", "c1_column": "_column", "column": "column"}


def test_rebase_path_basic():
    result = rebase_path(
        "/data/audio/folder1/file.wav", "/data/audio", "/output/waveforms"
    )
    assert result == "/output/waveforms/folder1/file.wav"


def test_rebase_path_with_s3_uri():
    result = rebase_path(
        "s3://bucket/data/audio/folder/file.wav",
        "data/audio",
        "s3://output-bucket/waveforms",
    )
    assert result == "s3://output-bucket/waveforms/folder/file.wav"


def test_rebase_path_mixed_uri_schemes():
    result = rebase_path(
        "/local/data/audio/file.mp3", "/local/data/audio", "s3://bucket/output"
    )
    assert result == "s3://bucket/output/file.mp3"


def test_rebase_path_with_suffix():
    result = rebase_path(
        "/data/audio/file.wav", "/data/audio", "/output", suffix="_processed"
    )
    assert result == "/output/file_processed.wav"


def test_rebase_path_with_extension_change():
    result = rebase_path("/data/audio/file.wav", "audio", "/output", extension="npy")
    assert result == "/output/file.npy"


def test_rebase_path_base_dir_not_in_path():
    with pytest.raises(
        ValueError, match="old_base '/data/audio' not found in src_path"
    ):
        rebase_path("/different/path/file.wav", "/data/audio", "/output")


def test_rebase_path_partial_match_base_dir():
    result = rebase_path("/home/user/data/audio/file.wav", "data/audio", "/output")
    assert result == "/output/file.wav"


def test_rebase_path_complex_s3_paths():
    result = rebase_path(
        "s3://bucket/balanced_train_segments/audio/folder/file.flac",
        "s3://bucket/balanced_train_segments",
        "s3://output-bucket/waveforms",
        suffix="_ch1",
        extension="npy",
    )
    assert result == "s3://output-bucket/waveforms/audio/folder/file_ch1.npy"


def test_rebase_path_file_without_extension():
    result = rebase_path("/data/audio/file_no_ext", "/data/audio", "/output")
    assert result == "/output/file_no_ext"

    # With new extension
    result = rebase_path(
        "/data/audio/file_no_ext", "/data/audio", "/output", extension="txt"
    )
    assert result == "/output/file_no_ext.txt"
