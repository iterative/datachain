import re
from dataclasses import dataclass
from typing import Any, Optional

import tomli


class ScriptMetaParsingError(Exception):
    def __init__(self, message):
        super().__init__(message)


@dataclass
class ScriptMeta:
    """
    Class that is parsing inline script metadata to get some basic information for
    running datachain script like python version, dependencies, files etc.
    Inline script metadata must follow the format described in https://packaging.python.org/en/latest/specifications/inline-script-metadata/#inline-script-metadata.
    Example of script with inline metadata:
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

    python_version: Optional[str]
    dependencies: list[str]
    files: dict[str, str]
    params: dict[str, Any]
    num_workers: Optional[int] = None

    def __init__(
        self,
        python_version: Optional[str] = None,
        dependencies: Optional[list[str]] = None,
        files: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        num_workers: Optional[int] = None,
    ):
        self.python_version = python_version
        self.dependencies = dependencies or []
        self.files = files or {}
        self.params = params or {}
        self.num_workers = num_workers

    def get_param(self, name: str) -> Any:
        return self.params.get(name)

    def get_file(self, name: str) -> Any:
        return self.files.get(name)

    @staticmethod
    def read_inline_meta(script: str) -> Optional[dict]:
        """Converts inline script metadata to dict with all found data"""
        regex = (
            r"(?m)^# \/\/\/ (?P<type>[a-zA-Z0-9-]+)[ \t]*$[\r\n|\r|\n]"
            "(?P<content>(?:^#(?:| .*)$[\r\n|\r|\n])+)^# \\/\\/\\/[ \t]*$"
        )
        name = "script"
        matches = list(
            filter(lambda m: m.group("type") == name, re.finditer(regex, script))
        )
        if len(matches) > 1:
            raise ValueError(f"Multiple {name} blocks found")
        if len(matches) == 1:
            content = "".join(
                line[2:] if line.startswith("# ") else line[1:]
                for line in matches[0].group("content").splitlines(keepends=True)
            )
            return tomli.loads(content)
        return None

    @staticmethod
    def parse(script: str) -> Optional["ScriptMeta"]:
        """
        Method that is parsing inline script metadata from datachain script and
        instantiating ScriptMeta class with found data. If no inline metadata is
        found, it returns None
        """
        try:
            meta = ScriptMeta.read_inline_meta(script)
            if not meta:
                return None
            custom = meta.get("tools", {}).get("datachain", {})
            return ScriptMeta(
                python_version=meta.get("requires-python"),
                dependencies=meta.get("dependencies"),
                num_workers=custom.get("workers", {}).get("num_workers"),
                files=custom.get("files"),
                params=custom.get("params"),
            )
        except Exception as e:
            raise ScriptMetaParsingError(f"Error when parsing script meta: {e}") from e
