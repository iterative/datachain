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
        # Accept both JSON config (new) and legacy base64-serialized clone params
        try:
            data = json.loads(metastore_serialized)
        except json.JSONDecodeError as _exc:
            # Fallback to legacy serializer
            from datachain.data_storage.serializer import deserialize

            obj = deserialize(metastore_serialized)
            # Basic type safety: must be a metastore
            from datachain.data_storage import AbstractMetastore

            if not isinstance(obj, AbstractMetastore):
                raise RuntimeError(  # noqa: TRY004
                    "DATACHAIN__METASTORE must be an instance of AbstractMetastore"
                ) from _exc
            return obj
        else:
            if not (isinstance(data, dict) and data.get("kind") == "sqlite_metastore"):
                raise RuntimeError("Unsupported metastore config payload")
            return SQLiteMetastoreConfig.model_validate(data).build()

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
        # Accept both JSON config (new) and legacy base64-serialized clone params
        try:
            data = json.loads(warehouse_serialized)
        except json.JSONDecodeError as _exc:
            # Fallback to legacy serializer
            from datachain.data_storage.serializer import deserialize

            obj = deserialize(warehouse_serialized)
            from datachain.data_storage import AbstractWarehouse

            if not isinstance(obj, AbstractWarehouse):
                raise RuntimeError(  # noqa: TRY004
                    "DATACHAIN__WAREHOUSE must be an instance of AbstractWarehouse"
                ) from _exc
            return obj
        else:
            if not (isinstance(data, dict) and data.get("kind") == "sqlite_warehouse"):
                raise RuntimeError("Unsupported warehouse config payload")
            return SQLiteWarehouseConfig.model_validate(data).build()

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
