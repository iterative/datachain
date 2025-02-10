import re
from dataclasses import dataclass
from typing import Any, Optional

import tomllib


class ScriptMetaParsingError(Exception):
    def __init__(self, message):
        super().__init__(message)


@dataclass
class ScriptMeta:
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
    def read_inline_meta(script: str) -> dict | None:
        regex = (
            r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
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
            return tomllib.loads(content)
        return None

    @staticmethod
    def parse(script: str) -> Optional["ScriptMeta"]:
        try:
            meta = ScriptMeta.read_inline_meta(script)
            if not meta:
                return None
            return ScriptMeta(
                python_version=meta["requires-python"],
                dependencies=meta["dependencies"],
                num_workers=meta["tools"]["datachain"]["workers"]["num_workers"],
                files=meta["tools"]["datachain"]["files"],
                params=meta["tools"]["datachain"]["params"],
            )
        except Exception as e:
            raise ScriptMetaParsingError(f"Error when parsing script meta: {e}") from e
