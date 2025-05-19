def parse(version: str) -> tuple[int, int, int]:
    """Parsing semver into 3 integers: major, minor, patch"""
    validate(version)
    parts = version.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def validate(version: str) -> None:
    """
    Raises exception if version doesn't have valid semver format which is:
    <major>.<minor>.<patch> or one of version parts is not positive integer
    """
    error_message = (
        "Invalid version. It should be in format: <major>.<minor>.<patch> where"
        " each version part is positive integer"
    )
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(error_message)
    for part in parts:
        try:
            val = int(part)
            assert val >= 0
        except (ValueError, AssertionError):
            raise ValueError(error_message) from None


def create(major: int = 0, minor: int = 0, patch: int = 0) -> str:
    """Creates new semver from 3 integers: major, minor and patch"""
    if major < 0 or minor < 0 or patch < 0:
        raise ValueError("Major, minor and patch must be greater or equal to zero")

    return ".".join([str(major), str(minor), str(patch)])


def value(version: str) -> int:
    """
    Calculate integer value of a version. This is useful when comparing two versions
    """
    major, minor, patch = parse(version)
    return major * 100 + minor * 10 + patch


def compare(v1: str, v2: str) -> int:
    """
    Compares 2 versions and returns:
       -1 if v1 < v2
        0 if v1 == v2
        1 if v1 > v2
    """
    v1_val = value(v1)
    v2_val = value(v2)

    if v1_val < v2_val:
        return -1
    if v1_val > v2_val:
        return 1
    return 0
