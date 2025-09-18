from typing import Literal, Optional

from pydantic import BaseModel

from datachain.dataset import StorageURI

from .sqlite import (
    SQLiteDatabaseEngine,
    SQLiteMetastore,
    SQLiteWarehouse,
    get_db_file_in_memory,
)


class SQLiteDatabaseEngineConfig(BaseModel):
    kind: Literal["sqlite_db_engine"] = "sqlite_db_engine"
    db_file: Optional[str] = None
    in_memory: bool = False

    def build(self) -> SQLiteDatabaseEngine:
        return SQLiteDatabaseEngine.from_db_file(
            get_db_file_in_memory(self.db_file, self.in_memory)
        )

    @classmethod
    def from_instance(cls, db: SQLiteDatabaseEngine) -> "SQLiteDatabaseEngineConfig":
        # Ensure Path-like values are serialized as strings for JSON/env usage
        db_file = db.db_file
        return cls(
            db_file=str(db_file) if db_file is not None else None,
            in_memory=(db_file == ":memory:"),
        )


class SQLiteMetastoreConfig(BaseModel):
    kind: Literal["sqlite_metastore"] = "sqlite_metastore"
    uri: str = ""
    db: SQLiteDatabaseEngineConfig

    def build(self) -> SQLiteMetastore:
        return SQLiteMetastore(uri=StorageURI(self.uri), db=self.db.build())

    @classmethod
    def from_instance(cls, ms: SQLiteMetastore) -> "SQLiteMetastoreConfig":
        return cls(uri=str(ms.uri), db=SQLiteDatabaseEngineConfig.from_instance(ms.db))


class SQLiteWarehouseConfig(BaseModel):
    kind: Literal["sqlite_warehouse"] = "sqlite_warehouse"
    db: SQLiteDatabaseEngineConfig

    def build(self) -> SQLiteWarehouse:
        return SQLiteWarehouse(db=self.db.build())

    @classmethod
    def from_instance(cls, wh: SQLiteWarehouse) -> "SQLiteWarehouseConfig":
        return cls(db=SQLiteDatabaseEngineConfig.from_instance(wh.db))
