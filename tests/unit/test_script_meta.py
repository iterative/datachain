import pytest

from datachain.script_meta import ScriptConfig, ScriptConfigParsingError


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
# [tools.datachain.attachments]
# image1 = "s3://ldb-public/image1.jpg"
# file1 = "s3://ldb-public/file.pdf"
#
# [tools.datachain.params]
# min_length_sec = 1
# cache = false
#
# [tools.datachain.inputs]
# threshold = 0.5
# start_ds_name = "ds://start"
#
# [tools.datachain.outputs]
# result_dataset = "ds://res"
# result_dir = "/temp"
#
# ///
import sys
import pandas as pd

print(f"Python version: {sys.version_info}")
print(f"Pandas version: {pd.__version__}")
"""
    sm = ScriptConfig.parse(script)
    assert sm == ScriptConfig(
        python_version=">=3.12",
        dependencies=["pandas < 2.1.0", "numpy == 1.26.4"],
        attachments={
            "image1": "s3://ldb-public/image1.jpg",
            "file1": "s3://ldb-public/file.pdf",
        },
        params={"min_length_sec": "1", "cache": "False"},
        inputs={"threshold": 0.5, "start_ds_name": "ds://start"},
        outputs={"result_dataset": "ds://res", "result_dir": "/temp"},
        num_workers=3,
    )
    assert sm.get_param("non_existing", "default") == "default"


def test_parsing_no_metadata():
    script = """
import sys
import pandas as pd

print(f"Python version: {sys.version_info}")
print(f"Pandas version: {pd.__version__}")
"""

    assert ScriptConfig.parse(script) is None


def test_parsing_empty():
    script = """
# /// script
# ///
import sys
import pandas as pd

print(f"Python version: {sys.version_info}")
print(f"Pandas version: {pd.__version__}")
"""

    assert ScriptConfig.parse(script) is None


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
    assert ScriptConfig.parse(script) == ScriptConfig(
        python_version=">=3.12",
        dependencies=[],
        attachments={},
        params={},
        num_workers=None,
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
    with pytest.raises(ScriptConfigParsingError) as excinfo:
        ScriptConfig.parse(script)
    assert str(excinfo.value) == (
        "Error when parsing script meta: Invalid value (at line 1, column 17)"
    )
