import logging
from argparse import ArgumentParser, ArgumentTypeError

import pytest

from datachain.cli import (
    get_logging_level,
    get_parser,
)
from datachain.cli.parser.utils import find_columns_type
from datachain.cli.utils import CommaSeparatedArgs, KeyValueArgs


def test_find_columns_type():
    assert find_columns_type("") == ["path"]
    assert find_columns_type("du") == ["du"]
    assert find_columns_type("", default_colums_str="name") == ["name"]
    assert find_columns_type("du, name,PATH") == ["du", "name", "path"]

    with pytest.raises(ArgumentTypeError):
        find_columns_type("bogus")


def test_cli_parser():
    parser = get_parser()

    args = parser.parse_args(("ls", "s3://example-bucket/"))

    assert args.sources == ["s3://example-bucket/"]

    assert args.quiet == 0
    assert args.verbose == 0

    assert get_logging_level(args) == logging.INFO

    args = parser.parse_args(("ls", "s3://example-bucket/", "-vvv"))

    assert args.quiet == 0
    assert args.verbose == 3

    assert get_logging_level(args) == logging.DEBUG

    args = parser.parse_args(("ls", "s3://example-bucket/", "-q"))

    assert args.quiet == 1
    assert args.verbose == 0

    assert get_logging_level(args) == logging.CRITICAL


@pytest.mark.parametrize(
    "param,parsed",
    (
        ("p1", ["p1"]),
        ("p1,p2", ["p1", "p2"]),
    ),
)
def test_comma_separated_args(param, parsed):
    parser = ArgumentParser()
    parser.add_argument("--param", default=[], action=CommaSeparatedArgs)

    args = parser.parse_args(("--param", param))
    assert args.param == parsed


@pytest.mark.parametrize("param", (None, ""))
def test_comma_separated_args_error(param):
    parser = ArgumentParser()
    parser.add_argument("--param", default=[], action=CommaSeparatedArgs)

    cmd = ["--param"]
    if param:
        cmd.append(param)
    with pytest.raises(SystemExit):
        parser.parse_args(cmd)


@pytest.mark.parametrize(
    "params,parsed",
    (
        ([], None),
        (["p1=foo"], {"p1": "foo"}),
        (["p1=bar", "p2=baz"], {"p1": "bar", "p2": "baz"}),
        (["p1=foo", "p1=bar"], {"p1": "bar"}),
        (["p1=foo", "p1=bar"], {"p1": "bar"}),
    ),
)
def test_key_value_args(params, parsed):
    parser = ArgumentParser()
    parser.add_argument("--param", nargs=1, action=KeyValueArgs)

    cmd = []
    for p in params:
        cmd.extend(["--param", p])

    args = parser.parse_args(cmd)
    assert args.param == parsed


@pytest.mark.parametrize("param", (None, "p1", "=", "p1=", "=foo"))
def test_key_value_args_error(param):
    parser = ArgumentParser()
    parser.add_argument("--param", nargs=1, action=KeyValueArgs)

    cmd = ["--param"]
    if param:
        cmd.append(param)
    with pytest.raises(SystemExit):
        parser.parse_args(cmd)
