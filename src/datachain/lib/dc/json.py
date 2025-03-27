import os
import os.path
import re
from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
)

from datachain.lib.data_model import DataType
from datachain.lib.file import (
    File,
    FileType,
)
from datachain.lib.meta_formats import read_meta

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def from_json(
    path: Union[str, os.PathLike[str]],
    type: FileType = "text",
    spec: Optional[DataType] = None,
    schema_from: Optional[str] = "auto",
    jmespath: Optional[str] = None,
    object_name: Optional[str] = "",
    model_name: Optional[str] = None,
    format: Optional[str] = "json",
    nrows=None,
    **kwargs,
) -> "DataChain":
    """Get data from JSON. It returns the chain itself.

    Parameters:
        path : storage URI with directory. URI must start with storage prefix such
            as `s3://`, `gs://`, `az://` or "file:///"
        type : read file as "binary", "text", or "image" data. Default is "text".
        spec : optional Data Model
        schema_from : path to sample to infer spec (if schema not provided)
        object_name : generated object column name
        model_name : optional generated model name
        format: "json", "jsonl"
        jmespath : optional JMESPATH expression to reduce JSON
        nrows : optional row limit for jsonl and JSON arrays

    Example:
        infer JSON schema from data, reduce using JMESPATH
        ```py
        import datachain as dc
        chain = dc.from_json("gs://json", jmespath="key1.key2")
        ```

        infer JSON schema from a particular path
        ```py
        import datachain as dc
        chain = dc.from_json("gs://json_ds", schema_from="gs://json/my.json")
        ```
    """
    from .storage import from_storage

    if schema_from == "auto":
        schema_from = os.fspath(path)

    def jmespath_to_name(s: str):
        name_end = re.search(r"\W", s).start() if re.search(r"\W", s) else len(s)  # type: ignore[union-attr]
        return s[:name_end]

    if (not object_name) and jmespath:
        object_name = jmespath_to_name(jmespath)
    if not object_name:
        object_name = format
    chain = from_storage(uri=path, type=type, **kwargs)
    signal_dict = {
        object_name: read_meta(
            schema_from=schema_from,
            format=format,
            spec=spec,
            model_name=model_name,
            jmespath=jmespath,
            nrows=nrows,
        ),
        "params": {"file": File},
    }
    # disable prefetch if nrows is set
    settings = {"prefetch": 0} if nrows else {}
    return chain.settings(**settings).gen(**signal_dict)  # type: ignore[misc, arg-type]
