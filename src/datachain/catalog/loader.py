import json
import os
import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any, Optional

from datachain.utils import get_envs_by_prefix

if TYPE_CHECKING:
    from datachain.catalog import Catalog
    from datachain.data_storage import AbstractMetastore, AbstractWarehouse
    from datachain.query.udf import AbstractUDFDistributor

METASTORE_SERIALIZED = "DATACHAIN__METASTORE"
METASTORE_IMPORT_PATH = "DATACHAIN_METASTORE"
METASTORE_ARG_PREFIX = "DATACHAIN_METASTORE_ARG_"
WAREHOUSE_SERIALIZED = "DATACHAIN__WAREHOUSE"
WAREHOUSE_IMPORT_PATH = "DATACHAIN_WAREHOUSE"
WAREHOUSE_ARG_PREFIX = "DATACHAIN_WAREHOUSE_ARG_"
DISTRIBUTED_IMPORT_PYTHONPATH = "DATACHAIN_DISTRIBUTED_PYTHONPATH"
DISTRIBUTED_IMPORT_PATH = "DATACHAIN_DISTRIBUTED"
DISTRIBUTED_DISABLED = "DATACHAIN_DISTRIBUTED_DISABLED"

IN_MEMORY_ERROR_MESSAGE = "In-memory is only supported on SQLite"


def get_metastore(in_memory: bool = False) -> "AbstractMetastore":
    from datachain.data_storage.config import (
        SQLiteDatabaseEngineConfig,
        SQLiteMetastoreConfig,
    )

    metastore_serialized = os.environ.get(METASTORE_SERIALIZED)
    if metastore_serialized:
        data = json.loads(metastore_serialized)
        if not isinstance(data, dict) or "kind" not in data:
            raise RuntimeError("Unsupported metastore config payload")
        kind = data.get("kind")
        if kind == "sqlite_metastore":
            return SQLiteMetastoreConfig.model_validate(data).build()
        if kind in {"api_metastore", "postgresql_metastore"}:
            try:
                from datachain_server.config import (
                    APIMetastoreConfig,
                    PostgreSQLMetastoreConfig,
                )
                cfg_map = {
                    "api_metastore": APIMetastoreConfig,
                    "postgresql_metastore": PostgreSQLMetastoreConfig,
                }
                cfg_cls = cfg_map[kind]
                return cfg_cls.model_validate(data).build()
            except Exception as exc:
                raise RuntimeError(
                    f"Metastore kind '{kind}' requires Studio runtime present"
                ) from exc
        raise RuntimeError(f"Unsupported metastore config payload kind: {kind}")

    metastore_import_path = os.environ.get(METASTORE_IMPORT_PATH)
    metastore_arg_envs = get_envs_by_prefix(METASTORE_ARG_PREFIX)
    # Convert env variable names to keyword argument names by lowercasing them
    metastore_args: dict[str, Any] = {
        k.lower(): v for k, v in metastore_arg_envs.items()
    }

    if not metastore_import_path:
        # Use config builder for default SQLite metastore
        db_cfg = SQLiteDatabaseEngineConfig(
            db_file=metastore_args.get("db_file"),
            in_memory=in_memory or bool(metastore_args.get("in_memory")),
        )
        ms_cfg = SQLiteMetastoreConfig(
            uri=str(metastore_args.get("uri", "")),
            db=db_cfg,
        )
        return ms_cfg.build()
    if in_memory:
        raise RuntimeError(IN_MEMORY_ERROR_MESSAGE)
    # Metastore paths are specified as (for example):
    # datachain.data_storage.SQLiteMetastore
    if "." not in metastore_import_path:
        raise RuntimeError(
            f"Invalid {METASTORE_IMPORT_PATH} import path: {metastore_import_path}"
        )
    module_name, _, class_name = metastore_import_path.rpartition(".")
    metastore = import_module(module_name)
    metastore_class = getattr(metastore, class_name)
    return metastore_class(**metastore_args)


def get_warehouse(in_memory: bool = False) -> "AbstractWarehouse":
    from datachain.data_storage.config import (
        SQLiteDatabaseEngineConfig,
        SQLiteWarehouseConfig,
    )

    warehouse_serialized = os.environ.get(WAREHOUSE_SERIALIZED)
    if warehouse_serialized:
        data = json.loads(warehouse_serialized)
        if not isinstance(data, dict) or "kind" not in data:
            raise RuntimeError("Unsupported warehouse config payload")
        kind = data.get("kind")
        if kind == "sqlite_warehouse":
            return SQLiteWarehouseConfig.model_validate(data).build()
        if kind in {"clickhouse_warehouse"}:
            try:
                from datachain_server.config import (
                    ClickHouseWarehouseConfig,
                )
                cfg_map = {"clickhouse_warehouse": ClickHouseWarehouseConfig}
                cfg_cls = cfg_map[kind]
                return cfg_cls.model_validate(data).build()
            except Exception as exc:
                raise RuntimeError(
                    f"Warehouse kind '{kind}' requires Studio runtime present"
                ) from exc
        raise RuntimeError(f"Unsupported warehouse config payload kind: {kind}")

    warehouse_import_path = os.environ.get(WAREHOUSE_IMPORT_PATH)
    warehouse_arg_envs = get_envs_by_prefix(WAREHOUSE_ARG_PREFIX)
    # Convert env variable names to keyword argument names by lowercasing them
    warehouse_args: dict[str, Any] = {
        k.lower(): v for k, v in warehouse_arg_envs.items()
    }

    if not warehouse_import_path:
        # Use config builder for default SQLite warehouse
        db_cfg = SQLiteDatabaseEngineConfig(
            db_file=warehouse_args.get("db_file"),
            in_memory=in_memory or bool(warehouse_args.get("in_memory")),
        )
        wh_cfg = SQLiteWarehouseConfig(db=db_cfg)
        return wh_cfg.build()
    if in_memory:
        raise RuntimeError(IN_MEMORY_ERROR_MESSAGE)
    # Warehouse paths are specified as (for example):
    # datachain.data_storage.SQLiteWarehouse
    if "." not in warehouse_import_path:
        raise RuntimeError(
            f"Invalid {WAREHOUSE_IMPORT_PATH} import path: {warehouse_import_path}"
        )
    module_name, _, class_name = warehouse_import_path.rpartition(".")
    warehouse = import_module(module_name)
    warehouse_class = getattr(warehouse, class_name)
    return warehouse_class(**warehouse_args)


def get_udf_distributor_class() -> Optional[type["AbstractUDFDistributor"]]:
    if os.environ.get(DISTRIBUTED_DISABLED) == "True":
        return None

    if not (distributed_import_path := os.environ.get(DISTRIBUTED_IMPORT_PATH)):
        return None

    # Distributed class paths are specified as (for example): module.classname
    if "." not in distributed_import_path:
        raise RuntimeError(
            f"Invalid {DISTRIBUTED_IMPORT_PATH} import path: {distributed_import_path}"
        )

    # Optional: set the Python path to look for the module
    distributed_import_pythonpath = os.environ.get(DISTRIBUTED_IMPORT_PYTHONPATH)
    if distributed_import_pythonpath and distributed_import_pythonpath not in sys.path:
        sys.path.insert(0, distributed_import_pythonpath)

    module_name, _, class_name = distributed_import_path.rpartition(".")
    distributed = import_module(module_name)
    return getattr(distributed, class_name)


def get_catalog(
    client_config: Optional[dict[str, Any]] = None,
    in_memory: bool = False,
) -> "Catalog":
    """
    Function that creates Catalog instance with appropriate metastore
    and warehouse classes. Metastore class can be provided with env variable
    DATACHAIN_METASTORE and if not provided, default one is used. Warehouse class
    can be provided with env variable DATACHAIN_WAREHOUSE and if not provided,

    If classes expects some kwargs, they can be provided via env variables
    by adding them with prefix (DATACHAIN_METASTORE_ARG_ and DATACHAIN_WAREHOUSE_ARG_)
    and name of variable after, e.g. if it accepts team_id as kwargs
    we can provide DATACHAIN_METASTORE_ARG_TEAM_ID=12345 env variable.
    """
    from datachain.catalog import Catalog

    metastore = get_metastore(in_memory=in_memory)
    return Catalog(
        metastore=metastore,
        warehouse=get_warehouse(in_memory=in_memory),
        client_config=client_config,
        in_memory=in_memory,
    )
