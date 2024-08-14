import os
from importlib import import_module
from typing import Any, Optional

from datachain.catalog import Catalog
from datachain.data_storage import (
    AbstractIDGenerator,
    AbstractMetastore,
    AbstractWarehouse,
)
from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import (
    SQLiteIDGenerator,
    SQLiteMetastore,
    SQLiteWarehouse,
)
from datachain.utils import get_envs_by_prefix

ID_GENERATOR_SERIALIZED = "DATACHAIN__ID_GENERATOR"
ID_GENERATOR_IMPORT_PATH = "DATACHAIN_ID_GENERATOR"
ID_GENERATOR_ARG_PREFIX = "DATACHAIN_ID_GENERATOR_ARG_"
METASTORE_SERIALIZED = "DATACHAIN__METASTORE"
METASTORE_IMPORT_PATH = "DATACHAIN_METASTORE"
METASTORE_ARG_PREFIX = "DATACHAIN_METASTORE_ARG_"
WAREHOUSE_SERIALIZED = "DATACHAIN__WAREHOUSE"
WAREHOUSE_IMPORT_PATH = "DATACHAIN_WAREHOUSE"
WAREHOUSE_ARG_PREFIX = "DATACHAIN_WAREHOUSE_ARG_"
DISTRIBUTED_IMPORT_PATH = "DATACHAIN_DISTRIBUTED"
DISTRIBUTED_ARG_PREFIX = "DATACHAIN_DISTRIBUTED_ARG_"

IN_MEMORY_ERROR_MESSAGE = "In-memory is only supported on SQLite"


def get_id_generator(in_memory: bool = False) -> "AbstractIDGenerator":
    id_generator_serialized = os.environ.get(ID_GENERATOR_SERIALIZED)
    if id_generator_serialized:
        id_generator_obj = deserialize(id_generator_serialized)
        if not isinstance(id_generator_obj, AbstractIDGenerator):
            raise RuntimeError(
                "Deserialized ID generator is not an instance of AbstractIDGenerator: "
                f"{id_generator_obj}"
            )
        return id_generator_obj

    id_generator_import_path = os.environ.get(ID_GENERATOR_IMPORT_PATH)
    id_generator_arg_envs = get_envs_by_prefix(ID_GENERATOR_ARG_PREFIX)
    # Convert env variable names to keyword argument names by lowercasing them
    id_generator_args: dict[str, Any] = {
        k.lower(): v for k, v in id_generator_arg_envs.items()
    }

    if not id_generator_import_path:
        id_generator_args["in_memory"] = in_memory
        return SQLiteIDGenerator(**id_generator_args)
    if in_memory:
        raise RuntimeError(IN_MEMORY_ERROR_MESSAGE)
    # ID generator paths are specified as (for example):
    # datachain.data_storage.SQLiteIDGenerator
    if "." not in id_generator_import_path:
        raise RuntimeError(
            f"Invalid {ID_GENERATOR_IMPORT_PATH} import path:"
            f"{id_generator_import_path}"
        )
    module_name, _, class_name = id_generator_import_path.rpartition(".")
    id_generator = import_module(module_name)
    id_generator_class = getattr(id_generator, class_name)
    return id_generator_class(**id_generator_args)


def get_metastore(
    id_generator: Optional["AbstractIDGenerator"], in_memory: bool = False
) -> "AbstractMetastore":
    metastore_serialized = os.environ.get(METASTORE_SERIALIZED)
    if metastore_serialized:
        metastore_obj = deserialize(metastore_serialized)
        if not isinstance(metastore_obj, AbstractMetastore):
            raise RuntimeError(
                "Deserialized Metastore is not an instance of AbstractMetastore: "
                f"{metastore_obj}"
            )
        return metastore_obj

    if id_generator is None:
        id_generator = get_id_generator()

    metastore_import_path = os.environ.get(METASTORE_IMPORT_PATH)
    metastore_arg_envs = get_envs_by_prefix(METASTORE_ARG_PREFIX)
    # Convert env variable names to keyword argument names by lowercasing them
    metastore_args: dict[str, Any] = {
        k.lower(): v for k, v in metastore_arg_envs.items()
    }

    if not metastore_import_path:
        if not isinstance(id_generator, SQLiteIDGenerator):
            raise ValueError("SQLiteMetastore can only be used with SQLiteIDGenerator")
        metastore_args["in_memory"] = in_memory
        return SQLiteMetastore(id_generator, **metastore_args)
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
    return metastore_class(id_generator, **metastore_args)


def get_warehouse(
    id_generator: Optional["AbstractIDGenerator"], in_memory: bool = False
) -> "AbstractWarehouse":
    warehouse_serialized = os.environ.get(WAREHOUSE_SERIALIZED)
    if warehouse_serialized:
        warehouse_obj = deserialize(warehouse_serialized)
        if not isinstance(warehouse_obj, AbstractWarehouse):
            raise RuntimeError(
                "Deserialized Warehouse is not an instance of AbstractWarehouse: "
                f"{warehouse_obj}"
            )
        return warehouse_obj

    if id_generator is None:
        id_generator = get_id_generator()

    warehouse_import_path = os.environ.get(WAREHOUSE_IMPORT_PATH)
    warehouse_arg_envs = get_envs_by_prefix(WAREHOUSE_ARG_PREFIX)
    # Convert env variable names to keyword argument names by lowercasing them
    warehouse_args: dict[str, Any] = {
        k.lower(): v for k, v in warehouse_arg_envs.items()
    }

    if not warehouse_import_path:
        if not isinstance(id_generator, SQLiteIDGenerator):
            raise ValueError("SQLiteWarehouse can only be used with SQLiteIDGenerator")
        warehouse_args["in_memory"] = in_memory
        return SQLiteWarehouse(id_generator, **warehouse_args)
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
    return warehouse_class(id_generator, **warehouse_args)


def get_distributed_class(**kwargs):
    distributed_import_path = os.environ.get(DISTRIBUTED_IMPORT_PATH)
    distributed_arg_envs = get_envs_by_prefix(DISTRIBUTED_ARG_PREFIX)
    # Convert env variable names to keyword argument names by lowercasing them
    distributed_args = {k.lower(): v for k, v in distributed_arg_envs.items()}

    if not distributed_import_path:
        raise RuntimeError(
            f"{DISTRIBUTED_IMPORT_PATH} import path is required "
            "for distributed UDF processing."
        )
    # Distributed class paths are specified as (for example):
    # module.classname
    if "." not in distributed_import_path:
        raise RuntimeError(
            f"Invalid {DISTRIBUTED_IMPORT_PATH} import path: {distributed_import_path}"
        )
    module_name, _, class_name = distributed_import_path.rpartition(".")
    distributed = import_module(module_name)
    distributed_class = getattr(distributed, class_name)
    return distributed_class(**distributed_args | kwargs)


def get_catalog(
    client_config: Optional[dict[str, Any]] = None, in_memory: bool = False
) -> Catalog:
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
    id_generator = get_id_generator(in_memory=in_memory)
    return Catalog(
        id_generator=id_generator,
        metastore=get_metastore(id_generator, in_memory=in_memory),
        warehouse=get_warehouse(id_generator, in_memory=in_memory),
        client_config=client_config,
        in_memory=in_memory,
    )
