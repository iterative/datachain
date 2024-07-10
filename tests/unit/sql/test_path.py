import posixpath as pp
import re

import pytest
from sqlalchemy import literal, select
from sqlalchemy.sql import func as f

from datachain.sql.functions import path as sql_path

PATHS = ["", "/", "name", "/name", "name/", "some/long/path"]
EXT_PATHS = [
    "",
    "abc.txt",
    "abc...txt",
    "abc",
    "abc/",
    "some/path/abc.tar.gz",
    "some/pa.th/abc",
]


def split_parent(path):
    parent, name = f"/{path}".rsplit("/", 1)
    return parent[1:], name


def file_stem(path):
    return pp.splitext(path)[0].rstrip(".")


def file_ext(path):
    return pp.splitext(path)[1].lstrip(".")


@pytest.mark.parametrize("func_base", [f.path, sql_path])
@pytest.mark.parametrize("func_name", ["parent", "name"])
def test_default_not_implement(func_base, func_name):
    """
    Importing datachain.sql.functions.path should register a custom compiler
    which raises an exception for these functions with the default
    SQLAlchemy dialect.
    """
    fn = getattr(func_base, func_name)
    expr = fn(literal("file:///some/file/path"))
    with pytest.raises(NotImplementedError, match=re.escape(f"path.{func_name}")):
        expr.compile()


@pytest.mark.parametrize("path", PATHS)
def test_parent(warehouse, path):
    query = select(f.path.parent(literal(path)))
    result = tuple(warehouse.db.execute(query))
    assert result == ((split_parent(path)[0],),)


@pytest.mark.parametrize("path", PATHS)
def test_name(warehouse, path):
    query = select(f.path.name(literal(path)))
    result = tuple(warehouse.db.execute(query))
    assert result == ((split_parent(path)[1],),)


@pytest.mark.parametrize("path", EXT_PATHS)
def test_file_stem(warehouse, path):
    query = select(sql_path.file_stem(literal(path)))
    result = tuple(warehouse.db.execute(query))
    assert result == ((file_stem(path),),)


@pytest.mark.parametrize("path", EXT_PATHS)
def test_file_ext(warehouse, path):
    query = select(sql_path.file_ext(literal(path)))
    result = tuple(warehouse.db.execute(query))
    assert result == ((file_ext(path),),)
