import pytest

from datachain.script_meta import ScriptMeta, ScriptMetaParsingError


def test_parsing_all_fields():
    script = """
# /// script
# requires-python = ">=3.12"
#
# dependencies = [
#   "pandas < 2.1.0",
#   "numpy == 1.26.4"
# ]
#
# [tools.datachain.workers]
# num_workers = 3
#
# [tools.datachain.files]
# image1 = "s3://ldb-public/image1.jpg"
# file1 = "s3://ldb-public/file.pdf"
#
# [tools.datachain.params]
# min_length_sec = 1
# cache = false
#
# ///
import sys
import pandas as pd

print(f"Python version: {sys.version_info}")
print(f"Pandas version: {pd.__version__}")
"""
    assert ScriptMeta.parse(script) == ScriptMeta(
        python_version=">=3.12",
        dependencies=["pandas < 2.1.0", "numpy == 1.26.4"],
        files={
            "image1": "s3://ldb-public/image1.jpg",
            "file1": "s3://ldb-public/file.pdf",
        },
        params={"min_length_sec": 1, "cache": False},
        num_workers=3,
    )


def test_parsing_no_metadata():
    script = """
import sys
import pandas as pd

print(f"Python version: {sys.version_info}")
print(f"Pandas version: {pd.__version__}")
"""

    assert ScriptMeta.parse(script) is None


def test_parsing_empty():
    script = """
# /// script
# ///
import sys
import pandas as pd

print(f"Python version: {sys.version_info}")
print(f"Pandas version: {pd.__version__}")
"""

    assert ScriptMeta.parse(script) is None


def test_parsing_only_python_version():
    script = """
# /// script
# requires-python = ">=3.12"
# ///
import sys
import pandas as pd

print(f"Python version: {sys.version_info}")
print(f"Pandas version: {pd.__version__}")
"""
    assert ScriptMeta.parse(script) == ScriptMeta(
        python_version=">=3.12", dependencies=[], files={}, params={}, num_workers=None
    )


def test_error_when_parsing():
    script = """
# /// script
# dependencies = [}
# ///
import sys
import pandas as pd

print(f"Python version: {sys.version_info}")
print(f"Pandas version: {pd.__version__}")
"""
    with pytest.raises(ScriptMetaParsingError) as excinfo:
        ScriptMeta.parse(script)
    assert str(excinfo.value) == (
        "Error when parsing script meta: Invalid value (at line 1, column 17)"
    )
