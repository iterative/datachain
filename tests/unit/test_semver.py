import pytest

from datachain import semver


def test_parse():
    assert semver.parse("0.1.2") == (0, 1, 2)


def test_parse_wrong_format():
    with pytest.raises(ValueError) as excinfo:
        semver.parse("1.2")
    assert str(excinfo.value) == (
        "Invalid version. It should be in format: <major>.<minor>.<patch> where"
        " each version part is positive integer"
    )


def test_create():
    assert semver.create() == "0.0.0"
    assert semver.create(1, 2, 3) == "1.2.3"


def test_create_wrong_values():
    with pytest.raises(ValueError) as excinfo:
        semver.create(-1, 2, 3)
    assert str(excinfo.value) == (
        "Major, minor and patch must be greater or equal to zero"
    )


def test_value():
    assert semver.value("0.0.0") == 0
    assert semver.value("1.2.3") == 123


@pytest.mark.parametrize(
    "v1,v2,result",
    [
        ("0.0.0", "0.0.0", 0),
        ("1.2.3", "1.2.3", 0),
        ("1.2.3", "1.2.4", -1),
        ("1.2.3", "1.2.2", 1),
    ],
)
def test_compare(v1, v2, result):
    assert semver.compare(v1, v2) == result


@pytest.mark.parametrize(
    "version,valid",
    [
        ("0.0.0", True),
        ("100.100.100", True),
        ("-1.2.3", False),
        ("1.2.3-alpha+01", False),
        ("1.2", False),
        ("1", False),
        ("1.2.3.4", False),
    ],
)
def test_validate(version, valid):
    if valid:
        semver.validate(version)
    else:
        with pytest.raises(ValueError) as excinfo:
            semver.validate(version)
        assert str(excinfo.value) == (
            "Invalid version. It should be in format: <major>.<minor>.<patch> where"
            " each version part is positive integer"
        )
