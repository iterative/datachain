import os
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "env_var,name,value",
    (
        ('{"foo":"bar"}', "foo", "bar"),
        ('{"foo":"bar2"}', "foo", "bar2"),
        ('{"foo":null}', "foo", None),
        ("{}", "foo", None),
        ("", "foo", None),
    ),
)
def test_query_param(env_var, name, value):
    with patch("datachain.query.params.params_cache", None):
        with patch.dict(os.environ, {"DATACHAIN_QUERY_PARAMS": env_var}):
            from datachain.query.params import param

            assert param(name) == value


@pytest.mark.parametrize("env_var", ('"foo"', "[1,2,3]", "null", "1", "true", "foo"))
def test_query_param_error(env_var):
    with patch("datachain.query.params.params_cache", None):
        with patch.dict(os.environ, {"DATACHAIN_QUERY_PARAMS": env_var}):
            from datachain.query.params import param

            with pytest.raises(ValueError):
                param("foo")


def test_query_param_key_error():
    from datachain.query.params import param

    with pytest.raises(TypeError):
        param(12)
