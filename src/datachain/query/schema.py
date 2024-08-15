import functools
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, Union

import attrs
import sqlalchemy as sa
from fsspec.callbacks import DEFAULT_CALLBACK, Callback

from datachain.sql.types import JSON, Boolean, DateTime, Int, Int64, SQLType, String

if TYPE_CHECKING:
    from datachain.catalog import Catalog
    from datachain.dataset import RowDict


DEFAULT_DELIMITER = "__"


class ColumnMeta(type):
    @staticmethod
    def to_db_name(name: str) -> str:
        return name.replace(".", DEFAULT_DELIMITER)

    def __getattr__(cls, name: str):
        return cls(ColumnMeta.to_db_name(name))


class Column(sa.ColumnClause, metaclass=ColumnMeta):
    inherit_cache: Optional[bool] = True

    def __init__(self, text, type_=None, is_literal=False, _selectable=None):
        """Dataset column."""
        self.name = ColumnMeta.to_db_name(text)
        super().__init__(
            self.name, type_=type_, is_literal=is_literal, _selectable=_selectable
        )

    def __getattr__(self, name: str):
        return Column(self.name + DEFAULT_DELIMITER + name)

    def glob(self, glob_str):
        """Search for matches using glob pattern matching."""
        return self.op("GLOB")(glob_str)

    def regexp(self, regexp_str):
        """Search for matches using regexp pattern matching."""
        return self.op("REGEXP")(regexp_str)


class UDFParameter(ABC):
    @abstractmethod
    def get_value(self, catalog: "Catalog", row: "RowDict", **kwargs) -> Any: ...

    async def get_value_async(
        self, catalog: "Catalog", row: "RowDict", mapper, **kwargs
    ) -> Any:
        return self.get_value(catalog, row, **kwargs)


@attrs.define(slots=False)
class ColumnParameter(UDFParameter):
    name: str

    def get_value(self, catalog, row, **kwargs):
        return row[self.name]


@attrs.define(slots=False)
class Object(UDFParameter):
    """
    Object is used as a placeholder parameter to indicate the actual stored object
    being passed as a parameter to the UDF.
    """

    reader: Callable

    def get_value(
        self,
        catalog: "Catalog",
        row: "RowDict",
        *,
        cache: bool = False,
        cb: Callback = DEFAULT_CALLBACK,
        **kwargs,
    ) -> Any:
        client = catalog.get_client(row["source"])
        uid = catalog._get_row_uid(row)
        if cache:
            client.download(uid, callback=cb)
        with client.open_object(uid, use_cache=cache, cb=cb) as f:
            return self.reader(f)

    async def get_value_async(
        self,
        catalog: "Catalog",
        row: "RowDict",
        mapper,
        *,
        cache: bool = False,
        cb: Callback = DEFAULT_CALLBACK,
        **kwargs,
    ) -> Any:
        client = catalog.get_client(row["source"])
        uid = catalog._get_row_uid(row)
        if cache:
            await client._download(uid, callback=cb)
        obj = await mapper.to_thread(
            functools.partial(client.open_object, uid, use_cache=cache, cb=cb)
        )
        with obj:
            return await mapper.to_thread(self.reader, obj)


@attrs.define(slots=False)
class Stream(UDFParameter):
    """
    A Stream() parameter receives a binary stream over the object contents.
    """

    def get_value(
        self,
        catalog: "Catalog",
        row: "RowDict",
        *,
        cache: bool = False,
        cb: Callback = DEFAULT_CALLBACK,
        **kwargs,
    ) -> Any:
        client = catalog.get_client(row["source"])
        uid = catalog._get_row_uid(row)
        if cache:
            client.download(uid, callback=cb)
        return client.open_object(uid, use_cache=cache, cb=cb)

    async def get_value_async(
        self,
        catalog: "Catalog",
        row: "RowDict",
        mapper,
        *,
        cache: bool = False,
        cb: Callback = DEFAULT_CALLBACK,
        **kwargs,
    ) -> Any:
        client = catalog.get_client(row["source"])
        uid = catalog._get_row_uid(row)
        if cache:
            await client._download(uid, callback=cb)
        return await mapper.to_thread(
            functools.partial(client.open_object, uid, use_cache=cache, cb=cb)
        )


@attrs.define(slots=False)
class LocalFilename(UDFParameter):
    """
    Placeholder parameter representing the local path to a cached copy of the object.

    If glob is None, then all files will be returned. If glob is specified,
    then only files matching the glob will be returned,
    otherwise None will be returned.
    """

    glob: Optional[str] = None

    def get_value(
        self,
        catalog: "Catalog",
        row: "RowDict",
        *,
        cb: Callback = DEFAULT_CALLBACK,
        **kwargs,
    ) -> Optional[str]:
        if self.glob and not fnmatch(row["name"], self.glob):  # type: ignore[type-var]
            # If the glob pattern is specified and the row filename
            # does not match it, then return None
            return None
        client = catalog.get_client(row["source"])
        uid = catalog._get_row_uid(row)
        client.download(uid, callback=cb)
        return client.cache.get_path(uid)

    async def get_value_async(
        self,
        catalog: "Catalog",
        row: "RowDict",
        mapper,
        *,
        cache: bool = False,
        cb: Callback = DEFAULT_CALLBACK,
        **kwargs,
    ) -> Optional[str]:
        if self.glob and not fnmatch(row["name"], self.glob):  # type: ignore[type-var]
            # If the glob pattern is specified and the row filename
            # does not match it, then return None
            return None
        client = catalog.get_client(row["source"])
        uid = catalog._get_row_uid(row)
        await client._download(uid, callback=cb)
        return client.cache.get_path(uid)


UDFParamSpec = Union[str, Column, UDFParameter]


def normalize_param(param: UDFParamSpec) -> UDFParameter:
    if isinstance(param, str):
        return ColumnParameter(param)
    if isinstance(param, Column):
        return ColumnParameter(param.name)
    if isinstance(param, UDFParameter):
        return param
    raise TypeError(f"Invalid UDF parameter: {param}")


class DatasetRow:
    schema: ClassVar[dict[str, type[SQLType]]] = {
        "source": String,
        "path": String,
        "size": Int64,
        "location": JSON,
        "vtype": String,
        "dir_type": Int,
        "owner_name": String,
        "owner_id": String,
        "is_latest": Boolean,
        "last_modified": DateTime,
        "version": String,
        "etag": String,
    }

    @staticmethod
    def create(
        path: str,
        source: str = "",
        size: int = 0,
        location: Optional[dict[str, Any]] = None,
        vtype: str = "",
        dir_type: int = 0,
        owner_name: str = "",
        owner_id: str = "",
        is_latest: bool = True,
        last_modified: Optional[datetime] = None,
        version: str = "",
        etag: str = "",
    ) -> tuple[
        str,
        str,
        int,
        Optional[str],
        str,
        int,
        str,
        str,
        bool,
        datetime,
        str,
        str,
        int,
    ]:
        if location:
            location = json.dumps([location])  # type: ignore [assignment]

        last_modified = last_modified or datetime.now(timezone.utc)

        return (  # type: ignore [return-value]
            source,
            path,
            size,
            location,
            vtype,
            dir_type,
            owner_name,
            owner_id,
            is_latest,
            last_modified,
            version,
            etag,
        )

    @staticmethod
    def extend(**columns):
        cols = {**DatasetRow.schema}
        cols.update(columns)
        return cols


C = Column
