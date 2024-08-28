import ast
import glob
import io
import json
import logging
import math
import os
import os.path
import posixpath
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import cached_property, reduce
from random import shuffle
from threading import Thread
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    NoReturn,
    Optional,
    Union,
)
from uuid import uuid4

import requests
import sqlalchemy as sa
import yaml
from sqlalchemy import Column
from tqdm import tqdm

from datachain.cache import DataChainCache, UniqueId
from datachain.client import Client
from datachain.config import get_remote_config, read_config
from datachain.dataset import (
    DATASET_PREFIX,
    QUERY_DATASET_PREFIX,
    DatasetDependency,
    DatasetRecord,
    DatasetStats,
    DatasetStatus,
    RowDict,
    create_dataset_uri,
    parse_dataset_uri,
)
from datachain.error import (
    ClientError,
    DataChainError,
    DatasetInvalidVersionError,
    DatasetNotFoundError,
    PendingIndexingError,
    QueryScriptCancelError,
    QueryScriptCompileError,
    QueryScriptDatasetNotFound,
    QueryScriptRunError,
)
from datachain.listing import Listing
from datachain.node import DirType, Node, NodeWithPath
from datachain.nodes_thread_pool import NodesThreadPool
from datachain.remote.studio import StudioClient
from datachain.sql.types import JSON, Boolean, DateTime, Int, Int64, SQLType, String
from datachain.storage import Storage, StorageStatus, StorageURI
from datachain.utils import (
    DataChainDir,
    batched,
    datachain_paths_join,
    import_object,
    parse_params_string,
)

from .datasource import DataSource
from .subclass import SubclassFinder

if TYPE_CHECKING:
    from datachain.data_storage import (
        AbstractIDGenerator,
        AbstractMetastore,
        AbstractWarehouse,
    )
    from datachain.dataset import DatasetVersion
    from datachain.job import Job

logger = logging.getLogger("datachain")

DEFAULT_DATASET_DIR = "dataset"
DATASET_FILE_SUFFIX = ".edatachain"
FEATURE_CLASSES = ["DataModel"]

TTL_INT = 4 * 60 * 60

INDEX_INTERNAL_ERROR_MESSAGE = "Internal error on indexing"
DATASET_INTERNAL_ERROR_MESSAGE = "Internal error on creating dataset"
# exit code we use if last statement in query script is not instance of DatasetQuery
QUERY_SCRIPT_INVALID_LAST_STATEMENT_EXIT_CODE = 10
# exit code we use if query script was canceled
QUERY_SCRIPT_CANCELED_EXIT_CODE = 11

# dataset pull
PULL_DATASET_MAX_THREADS = 10
PULL_DATASET_CHUNK_TIMEOUT = 3600
PULL_DATASET_SLEEP_INTERVAL = 0.1  # sleep time while waiting for chunk to be available
PULL_DATASET_CHECK_STATUS_INTERVAL = 20  # interval to check export status in Studio


def _raise_remote_error(error_message: str) -> NoReturn:
    raise DataChainError(f"Error from server: {error_message}")


def noop(_: str):
    pass


@contextmanager
def print_and_capture(
    stream: "IO[bytes]|IO[str]", callback: Callable[[str], None] = noop
) -> "Iterator[list[str]]":
    lines: list[str] = []
    append = lines.append

    def loop() -> None:
        buffer = b""
        while byt := stream.read(1):  # Read one byte at a time
            buffer += byt.encode("utf-8") if isinstance(byt, str) else byt

            if byt in (b"\n", b"\r"):  # Check for newline or carriage return
                line = buffer.decode("utf-8")
                print(line, end="")
                callback(line)
                append(line)
                buffer = b""  # Clear buffer for next line

        if buffer:  # Handle any remaining data in the buffer
            line = buffer.decode("utf-8")
            print(line, end="")
            callback(line)
            append(line)

    thread = Thread(target=loop, daemon=True)
    thread.start()

    try:
        yield lines
    finally:
        thread.join()


class QueryResult(NamedTuple):
    dataset: Optional[DatasetRecord]
    version: Optional[int]
    output: str
    preview: Optional[list[dict]]
    metrics: dict[str, Any]


class DatasetRowsFetcher(NodesThreadPool):
    def __init__(
        self,
        metastore: "AbstractMetastore",
        warehouse: "AbstractWarehouse",
        remote_config: dict[str, Any],
        dataset_name: str,
        dataset_version: int,
        schema: dict[str, Union[SQLType, type[SQLType]]],
        max_threads: int = PULL_DATASET_MAX_THREADS,
    ):
        super().__init__(max_threads)
        self._check_dependencies()
        self.metastore = metastore
        self.warehouse = warehouse
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.schema = schema
        self.last_status_check: Optional[float] = None

        self.studio_client = StudioClient(
            remote_config["url"], remote_config["username"], remote_config["token"]
        )

    def done_task(self, done):
        for task in done:
            task.result()

    def _check_dependencies(self) -> None:
        try:
            import lz4.frame  # noqa: F401
            import numpy as np  # noqa: F401
            import pandas as pd  # noqa: F401
            import pyarrow as pa  # noqa: F401
        except ImportError as exc:
            raise Exception(
                f"Missing dependency: {exc.name}\n"
                "To install run:\n"
                "\tpip install 'datachain[remote]'"
            ) from None

    def should_check_for_status(self) -> bool:
        if not self.last_status_check:
            return True
        return time.time() - self.last_status_check > PULL_DATASET_CHECK_STATUS_INTERVAL

    def check_for_status(self) -> None:
        """
        Method that checks export status in Studio and raises Exception if export
        failed or was removed.
        Checks are done every PULL_DATASET_CHECK_STATUS_INTERVAL seconds
        """
        export_status_response = self.studio_client.dataset_export_status(
            self.dataset_name, self.dataset_version
        )
        if not export_status_response.ok:
            _raise_remote_error(export_status_response.message)

        export_status = export_status_response.data["status"]  # type: ignore [index]

        if export_status == "failed":
            _raise_remote_error("Dataset export failed in Studio")
        if export_status == "removed":
            _raise_remote_error("Dataset export removed in Studio")

        self.last_status_check = time.time()

    def fix_columns(self, df) -> None:
        import pandas as pd

        """
        Method that does various column decoding or parsing, depending on a type
        before inserting into DB
        """
        # we get dataframe from parquet export files where datetimes are serialized
        # as timestamps so we need to parse it back to datetime objects
        for c in [c for c, t in self.schema.items() if t == DateTime]:
            df[c] = pd.to_datetime(df[c], unit="s")

        # strings are represented as binaries in parquet export so need to
        # decode it back to strings
        for c in [c for c, t in self.schema.items() if t == String]:
            df[c] = df[c].str.decode("utf-8")

    def do_task(self, urls):
        import lz4.frame
        import pandas as pd

        # metastore and warehouse are not thread safe
        with self.metastore.clone() as metastore, self.warehouse.clone() as warehouse:
            dataset = metastore.get_dataset(self.dataset_name)

            urls = list(urls)
            while urls:
                for url in urls:
                    if self.should_check_for_status():
                        self.check_for_status()

                    r = requests.get(url, timeout=PULL_DATASET_CHUNK_TIMEOUT)
                    if r.status_code == 404:
                        time.sleep(PULL_DATASET_SLEEP_INTERVAL)
                        # moving to the next url
                        continue

                    r.raise_for_status()

                    df = pd.read_parquet(io.BytesIO(lz4.frame.decompress(r.content)))

                    self.fix_columns(df)

                    # id will be autogenerated in DB
                    df = df.drop("sys__id", axis=1)

                    inserted = warehouse.insert_dataset_rows(
                        df, dataset, self.dataset_version
                    )
                    self.increase_counter(inserted)  # type: ignore [arg-type]
                    urls.remove(url)


@dataclass
class NodeGroup:
    """Class for a group of nodes from the same source"""

    listing: Listing
    sources: list[DataSource]

    # The source path within the bucket
    # (not including the bucket name or s3:// prefix)
    source_path: str = ""
    is_edatachain: bool = False
    dataset_name: Optional[str] = None
    dataset_version: Optional[int] = None
    instantiated_nodes: Optional[list[NodeWithPath]] = None

    @property
    def is_dataset(self) -> bool:
        return bool(self.dataset_name)

    def iternodes(self, recursive: bool = False):
        for src in self.sources:
            if recursive and src.is_container():
                for nwp in src.find():
                    yield nwp.n
            else:
                yield src.node

    def download(self, recursive: bool = False, pbar=None) -> None:
        """
        Download this node group to cache.
        """
        if self.sources:
            self.listing.client.fetch_nodes(
                self.iternodes(recursive), shared_progress_bar=pbar
            )


def check_output_dataset_file(
    output: str,
    force: bool = False,
    dataset_filename: Optional[str] = None,
    skip_check_edatachain: bool = False,
) -> str:
    """
    Checks the dataset filename for existence or if it should be force-overwritten.
    """
    dataset_file = (
        dataset_filename if dataset_filename else output + DATASET_FILE_SUFFIX
    )
    if not skip_check_edatachain and os.path.exists(dataset_file):
        if force:
            os.remove(dataset_file)
        else:
            raise RuntimeError(f"Output dataset file already exists: {dataset_file}")
    return dataset_file


def parse_edatachain_file(filename: str) -> list[dict[str, Any]]:
    with open(filename, encoding="utf-8") as f:
        contents = yaml.safe_load(f)

    if not isinstance(contents, list):
        contents = [contents]

    for entry in contents:
        if not isinstance(entry, dict):
            raise TypeError(
                "Failed parsing EDataChain file, "
                "each data source entry must be a dictionary"
            )
        if "data-source" not in entry or "files" not in entry:
            raise ValueError(
                "Failed parsing EDataChain file, "
                "each data source entry must contain the "
                '"data-source" and "files" keys'
            )

    return contents


def prepare_output_for_cp(
    node_groups: list[NodeGroup],
    output: str,
    force: bool = False,
    edatachain_only: bool = False,
    no_edatachain_file: bool = False,
) -> tuple[bool, Optional[str]]:
    total_node_count = 0
    for node_group in node_groups:
        if not node_group.sources:
            raise FileNotFoundError(
                f"No such file or directory: {node_group.source_path}"
            )
        total_node_count += len(node_group.sources)

    always_copy_dir_contents = False
    copy_to_filename = None

    if edatachain_only:
        return always_copy_dir_contents, copy_to_filename

    if not os.path.isdir(output):
        if all(n.is_dataset for n in node_groups):
            os.mkdir(output)
        elif total_node_count == 1:
            first_source = node_groups[0].sources[0]
            if first_source.is_container():
                if os.path.exists(output):
                    if force:
                        os.remove(output)
                    else:
                        raise FileExistsError(f"Path already exists: {output}")
                always_copy_dir_contents = True
                os.mkdir(output)
            else:  # Is a File
                if os.path.exists(output):
                    if force:
                        os.remove(output)
                    else:
                        raise FileExistsError(f"Path already exists: {output}")
                copy_to_filename = output
        else:
            raise FileNotFoundError(f"Is not a directory: {output}")

    if copy_to_filename and not no_edatachain_file:
        raise RuntimeError("File to file cp not supported with .edatachain files!")

    return always_copy_dir_contents, copy_to_filename


def collect_nodes_for_cp(
    node_groups: Iterable[NodeGroup],
    recursive: bool = False,
) -> tuple[int, int]:
    total_size: int = 0
    total_files: int = 0

    # Collect all sources to process
    for node_group in node_groups:
        listing: Listing = node_group.listing
        valid_sources: list[DataSource] = []
        for dsrc in node_group.sources:
            if dsrc.is_single_object():
                total_size += dsrc.node.size
                total_files += 1
                valid_sources.append(dsrc)
            else:
                node = dsrc.node
                if not recursive:
                    print(f"{node.full_path} is a directory (not copied).")
                    continue
                add_size, add_files = listing.du(node, count_files=True)
                total_size += add_size
                total_files += add_files
                valid_sources.append(dsrc)

        node_group.sources = valid_sources

    return total_size, total_files


def get_download_bar(bar_format: str, total_size: int):
    return tqdm(
        desc="Downloading files: ",
        unit="B",
        bar_format=bar_format,
        unit_scale=True,
        unit_divisor=1000,
        total=total_size,
    )


def instantiate_node_groups(
    node_groups: Iterable[NodeGroup],
    output: str,
    bar_format: str,
    total_files: int,
    force: bool = False,
    recursive: bool = False,
    virtual_only: bool = False,
    always_copy_dir_contents: bool = False,
    copy_to_filename: Optional[str] = None,
) -> None:
    instantiate_progress_bar = (
        None
        if virtual_only
        else tqdm(
            desc=f"Instantiating {output}: ",
            unit=" f",
            bar_format=bar_format,
            unit_scale=True,
            unit_divisor=1000,
            total=total_files,
        )
    )

    output_dir = output
    if copy_to_filename:
        output_dir = os.path.dirname(output)
        if not output_dir:
            output_dir = "."

    # Instantiate these nodes
    for node_group in node_groups:
        if not node_group.sources:
            continue
        listing: Listing = node_group.listing
        source_path: str = node_group.source_path

        copy_dir_contents = always_copy_dir_contents or source_path.endswith("/")
        instantiated_nodes = listing.collect_nodes_to_instantiate(
            node_group.sources,
            copy_to_filename,
            recursive,
            copy_dir_contents,
            source_path,
            node_group.is_edatachain,
            node_group.is_dataset,
        )
        if not virtual_only:
            listing.instantiate_nodes(
                instantiated_nodes,
                output_dir,
                total_files,
                force=force,
                shared_progress_bar=instantiate_progress_bar,
            )
        node_group.instantiated_nodes = instantiated_nodes
    if instantiate_progress_bar:
        instantiate_progress_bar.close()


def compute_metafile_data(node_groups) -> list[dict[str, Any]]:
    metafile_data = []
    for node_group in node_groups:
        if not node_group.sources:
            continue
        listing: Listing = node_group.listing
        source_path: str = node_group.source_path
        if not node_group.is_dataset:
            assert listing.storage
            data_source = listing.storage.to_dict(source_path)
        else:
            data_source = {"uri": listing.metastore.uri}

        metafile_group = {"data-source": data_source, "files": []}
        for node in node_group.instantiated_nodes:
            if not node.n.is_dir:
                metafile_group["files"].append(node.get_metafile_data())
        if metafile_group["files"]:
            metafile_data.append(metafile_group)

    return metafile_data


def find_column_to_str(  # noqa: PLR0911
    row: tuple[Any, ...], field_lookup: dict[str, int], src: DataSource, column: str
) -> str:
    if column == "du":
        return str(
            src.listing.du(
                {f: row[field_lookup[f]] for f in ["dir_type", "size", "path"]}
            )[0]
        )
    if column == "name":
        return posixpath.basename(row[field_lookup["path"]]) or ""
    if column == "owner":
        return row[field_lookup["owner_name"]] or ""
    if column == "path":
        is_dir = row[field_lookup["dir_type"]] == DirType.DIR
        path = row[field_lookup["path"]]
        if is_dir and path:
            full_path = path + "/"
        else:
            full_path = path
        return src.get_node_full_path_from_path(full_path)
    if column == "size":
        return str(row[field_lookup["size"]])
    if column == "type":
        dt = row[field_lookup["dir_type"]]
        if dt == DirType.DIR:
            return "d"
        if dt == DirType.FILE:
            return "f"
        if dt == DirType.TAR_ARCHIVE:
            return "t"
        # Unknown - this only happens if a type was added elsewhere but not here
        return "u"
    return ""


def form_module_source(source_ast):
    module = ast.Module(body=source_ast, type_ignores=[])
    module = ast.fix_missing_locations(module)
    return ast.unparse(module)


class Catalog:
    def __init__(
        self,
        id_generator: "AbstractIDGenerator",
        metastore: "AbstractMetastore",
        warehouse: "AbstractWarehouse",
        cache_dir=None,
        tmp_dir=None,
        client_config: Optional[dict[str, Any]] = None,
        warehouse_ready_callback: Optional[
            Callable[["AbstractWarehouse"], None]
        ] = None,
        in_memory: bool = False,
    ):
        datachain_dir = DataChainDir(cache=cache_dir, tmp=tmp_dir)
        datachain_dir.init()
        self.id_generator = id_generator
        self.metastore = metastore
        self._warehouse = warehouse
        self.cache = DataChainCache(datachain_dir.cache, datachain_dir.tmp)
        self.client_config = client_config if client_config is not None else {}
        self._init_params = {
            "cache_dir": cache_dir,
            "tmp_dir": tmp_dir,
        }
        self._warehouse_ready_callback = warehouse_ready_callback
        self.in_memory = in_memory

    @cached_property
    def warehouse(self) -> "AbstractWarehouse":
        if self._warehouse_ready_callback:
            self._warehouse_ready_callback(self._warehouse)

        return self._warehouse

    def get_init_params(self) -> dict[str, Any]:
        return {
            **self._init_params,
            "client_config": self.client_config,
        }

    def copy(self, cache=True, db=True):
        result = copy(self)
        if not db:
            result.id_generator = None
            result.metastore = None
            result._warehouse = None
            result.warehouse = None
        return result

    @classmethod
    def generate_query_dataset_name(cls) -> str:
        return f"{QUERY_DATASET_PREFIX}_{uuid4().hex}"

    def attach_query_wrapper(self, code_ast):
        if code_ast.body:
            last_expr = code_ast.body[-1]
            if isinstance(last_expr, ast.Expr):
                new_expressions = [
                    ast.Import(
                        names=[ast.alias(name="datachain.query.dataset", asname=None)]
                    ),
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Attribute(
                                        value=ast.Name(id="datachain", ctx=ast.Load()),
                                        attr="query",
                                        ctx=ast.Load(),
                                    ),
                                    attr="dataset",
                                    ctx=ast.Load(),
                                ),
                                attr="query_wrapper",
                                ctx=ast.Load(),
                            ),
                            args=[last_expr],
                            keywords=[],
                        )
                    ),
                ]
                code_ast.body[-1:] = new_expressions
            else:
                raise Exception("Last line in a script was not an expression")
        return code_ast

    def compile_query_script(
        self, script: str, feature_module_name: str
    ) -> tuple[Union[str, None], str]:
        code_ast = ast.parse(script)
        code_ast = self.attach_query_wrapper(code_ast)
        finder = SubclassFinder(FEATURE_CLASSES)
        finder.visit(code_ast)

        if not finder.feature_class:
            main_module = form_module_source([*finder.imports, *finder.main_body])
            return None, main_module

        feature_import = ast.ImportFrom(
            module=feature_module_name,
            names=[ast.alias(name="*", asname=None)],
            level=0,
        )
        feature_module = form_module_source([*finder.imports, *finder.feature_class])
        main_module = form_module_source(
            [*finder.imports, feature_import, *finder.main_body]
        )

        return feature_module, main_module

    def parse_url(self, uri: str, **config: Any) -> tuple[Client, str]:
        config = config or self.client_config
        return Client.parse_url(uri, self.cache, **config)

    def get_client(self, uri: StorageURI, **config: Any) -> Client:
        """
        Return the client corresponding to the given source `uri`.
        """
        config = config or self.client_config
        cls = Client.get_implementation(uri)
        return cls.from_source(uri, self.cache, **config)

    def enlist_source(
        self,
        source: str,
        ttl: int,
        force_update=False,
        skip_indexing=False,
        client_config=None,
    ) -> tuple[Listing, str]:
        if force_update and skip_indexing:
            raise ValueError(
                "Both force_update and skip_indexing flags"
                " cannot be True at the same time"
            )

        partial_id: Optional[int]
        partial_path: Optional[str]

        client_config = client_config or self.client_config
        client, path = self.parse_url(source, **client_config)
        stem = os.path.basename(os.path.normpath(path))
        prefix = (
            posixpath.dirname(path)
            if glob.has_magic(stem) or client.fs.isfile(source)
            else path
        )
        storage_dataset_name = Storage.dataset_name(
            client.uri, posixpath.join(prefix, "")
        )
        source_metastore = self.metastore.clone(client.uri)

        columns = [
            Column("vtype", String),
            Column("dir_type", Int),
            Column("path", String),
            Column("etag", String),
            Column("version", String),
            Column("is_latest", Boolean),
            Column("last_modified", DateTime(timezone=True)),
            Column("size", Int64),
            Column("owner_name", String),
            Column("owner_id", String),
            Column("location", JSON),
            Column("source", String),
        ]

        if skip_indexing:
            source_metastore.create_storage_if_not_registered(client.uri)
            storage = source_metastore.get_storage(client.uri)
            source_metastore.init_partial_id(client.uri)
            partial_id = source_metastore.get_next_partial_id(client.uri)

            source_metastore = self.metastore.clone(
                uri=client.uri, partial_id=partial_id
            )
            source_metastore.init(client.uri)

            source_warehouse = self.warehouse.clone()
            dataset = self.create_dataset(
                storage_dataset_name, columns=columns, listing=True
            )

            return (
                Listing(storage, source_metastore, source_warehouse, client, dataset),
                path,
            )

        (
            storage,
            need_index,
            in_progress,
            partial_id,
            partial_path,
        ) = source_metastore.register_storage_for_indexing(
            client.uri, force_update, prefix
        )
        if in_progress:
            raise PendingIndexingError(f"Pending indexing operation: uri={storage.uri}")

        if not need_index:
            assert partial_id is not None
            assert partial_path is not None
            source_metastore = self.metastore.clone(
                uri=client.uri, partial_id=partial_id
            )
            source_warehouse = self.warehouse.clone()
            dataset = self.get_dataset(Storage.dataset_name(client.uri, partial_path))
            lst = Listing(storage, source_metastore, source_warehouse, client, dataset)
            logger.debug(
                "Using cached listing %s. Valid till: %s",
                storage.uri,
                storage.expires_to_local,
            )
            # Listing has to have correct version of data storage
            # initialized with correct Storage

            self.update_dataset_version_with_warehouse_info(
                dataset,
                dataset.latest_version,
            )

            return lst, path

        source_metastore.init_partial_id(client.uri)
        partial_id = source_metastore.get_next_partial_id(client.uri)

        source_metastore.init(client.uri)
        source_metastore = self.metastore.clone(uri=client.uri, partial_id=partial_id)

        source_warehouse = self.warehouse.clone()

        dataset = self.create_dataset(
            storage_dataset_name, columns=columns, listing=True
        )

        lst = Listing(storage, source_metastore, source_warehouse, client, dataset)

        try:
            lst.fetch(prefix)

            source_metastore.mark_storage_indexed(
                storage.uri,
                StorageStatus.PARTIAL if prefix else StorageStatus.COMPLETE,
                ttl,
                prefix=prefix,
                partial_id=partial_id,
                dataset=dataset,
            )

            self.update_dataset_version_with_warehouse_info(
                dataset,
                dataset.latest_version,
            )

        except ClientError as e:
            # for handling cloud errors
            error_message = INDEX_INTERNAL_ERROR_MESSAGE
            if e.error_code in ["InvalidAccessKeyId", "SignatureDoesNotMatch"]:
                error_message = "Invalid cloud credentials"

            source_metastore.mark_storage_indexed(
                storage.uri,
                StorageStatus.FAILED,
                ttl,
                prefix=prefix,
                error_message=error_message,
                error_stack=traceback.format_exc(),
                dataset=dataset,
            )
            self._remove_dataset_rows_and_warehouse_info(
                dataset, dataset.latest_version
            )
            raise
        except:
            source_metastore.mark_storage_indexed(
                storage.uri,
                StorageStatus.FAILED,
                ttl,
                prefix=prefix,
                error_message=INDEX_INTERNAL_ERROR_MESSAGE,
                error_stack=traceback.format_exc(),
                dataset=dataset,
            )
            self._remove_dataset_rows_and_warehouse_info(
                dataset, dataset.latest_version
            )
            raise

        lst.storage = storage

        return lst, path

    def _remove_dataset_rows_and_warehouse_info(
        self, dataset: DatasetRecord, version: int, **kwargs
    ):
        self.warehouse.drop_dataset_rows_table(dataset, version)
        self.update_dataset_version_with_warehouse_info(
            dataset,
            version,
            rows_dropped=True,
            **kwargs,
        )

    def enlist_sources(
        self,
        sources: list[str],
        ttl: int,
        update: bool,
        skip_indexing=False,
        client_config=None,
        only_index=False,
    ) -> Optional[list["DataSource"]]:
        enlisted_sources = []
        for src in sources:  # Opt: parallel
            listing, file_path = self.enlist_source(
                src,
                ttl,
                update,
                skip_indexing=skip_indexing,
                client_config=client_config or self.client_config,
            )
            enlisted_sources.append((listing, file_path))

        if only_index:
            # sometimes we don't really need listing result (e.g on indexing process)
            # so this is to improve performance
            return None

        dsrc_all: list[DataSource] = []
        for listing, file_path in enlisted_sources:
            nodes = listing.expand_path(file_path)
            dir_only = file_path.endswith("/")
            dsrc_all.extend(DataSource(listing, node, dir_only) for node in nodes)
        return dsrc_all

    def enlist_sources_grouped(
        self,
        sources: list[str],
        ttl: int,
        update: bool,
        no_glob: bool = False,
        client_config=None,
    ) -> list[NodeGroup]:
        from datachain.query import DatasetQuery

        def _row_to_node(d: dict[str, Any]) -> Node:
            del d["source"]
            return Node.from_dict(d)

        enlisted_sources: list[tuple[bool, bool, Any]] = []
        client_config = client_config or self.client_config
        for src in sources:  # Opt: parallel
            if src.endswith(DATASET_FILE_SUFFIX) and os.path.isfile(src):
                # TODO: Also allow using EDataChain files from cloud locations?
                edatachain_data = parse_edatachain_file(src)
                indexed_sources = []
                for ds in edatachain_data:
                    listing, source_path = self.enlist_source(
                        ds["data-source"]["uri"],
                        ttl,
                        update,
                        client_config=client_config,
                    )
                    paths = datachain_paths_join(
                        source_path, (f["name"] for f in ds["files"])
                    )
                    indexed_sources.append((listing, source_path, paths))
                enlisted_sources.append((True, False, indexed_sources))
            elif src.startswith("ds://"):
                ds_name, ds_version = parse_dataset_uri(src)
                dataset = self.get_dataset(ds_name)
                if not ds_version:
                    ds_version = dataset.latest_version
                dataset_sources = self.warehouse.get_dataset_sources(
                    dataset,
                    ds_version,
                )
                indexed_sources = []
                for source in dataset_sources:
                    client = self.get_client(source, **client_config)
                    uri = client.uri
                    ms = self.metastore.clone(uri, None)
                    st = self.warehouse.clone()
                    listing = Listing(None, ms, st, client, None)
                    rows = DatasetQuery(
                        name=dataset.name, version=ds_version, catalog=self
                    ).to_db_records()
                    indexed_sources.append(
                        (
                            listing,
                            source,
                            [_row_to_node(r) for r in rows],
                            ds_name,
                            ds_version,
                        )  # type: ignore [arg-type]
                    )

                enlisted_sources.append((False, True, indexed_sources))
            else:
                listing, source_path = self.enlist_source(
                    src, ttl, update, client_config=client_config
                )
                enlisted_sources.append((False, False, (listing, source_path)))

        node_groups = []
        for is_datachain, is_dataset, payload in enlisted_sources:  # Opt: parallel
            if is_dataset:
                for (
                    listing,
                    source_path,
                    nodes,
                    dataset_name,
                    dataset_version,
                ) in payload:
                    dsrc = [DataSource(listing, node) for node in nodes]
                    node_groups.append(
                        NodeGroup(
                            listing,
                            dsrc,
                            source_path,
                            dataset_name=dataset_name,
                            dataset_version=dataset_version,
                        )
                    )
            elif is_datachain:
                for listing, source_path, paths in payload:
                    dsrc = [DataSource(listing, listing.resolve_path(p)) for p in paths]
                    node_groups.append(
                        NodeGroup(listing, dsrc, source_path, is_edatachain=True)
                    )
            else:
                listing, source_path = payload
                as_container = source_path.endswith("/")
                dsrc = [
                    DataSource(listing, n, as_container)
                    for n in listing.expand_path(source_path, use_glob=not no_glob)
                ]
                node_groups.append(NodeGroup(listing, dsrc, source_path))

        return node_groups

    def unlist_source(self, uri: StorageURI) -> None:
        self.metastore.clone(uri=uri).mark_storage_not_indexed(uri)

    def storage_stats(self, uri: StorageURI) -> Optional[DatasetStats]:
        """
        Returns tuple with storage stats: total number of rows and total dataset size.
        """
        partial_path = self.metastore.get_last_partial_path(uri)
        if partial_path is None:
            return None
        dataset = self.get_dataset(Storage.dataset_name(uri, partial_path))

        return self.dataset_stats(dataset.name, dataset.latest_version)

    def create_dataset(
        self,
        name: str,
        version: Optional[int] = None,
        *,
        columns: Sequence[Column],
        feature_schema: Optional[dict] = None,
        query_script: str = "",
        create_rows: Optional[bool] = True,
        validate_version: Optional[bool] = True,
        listing: Optional[bool] = False,
    ) -> "DatasetRecord":
        """
        Creates new dataset of a specific version.
        If dataset is not yet created, it will create it with version 1
        If version is None, then next unused version is created.
        If version is given, then it must be an unused version number.
        """
        assert [c.name for c in columns if c.name != "sys__id"], f"got {columns=}"
        if not listing and Client.is_data_source_uri(name):
            raise RuntimeError(
                "Cannot create dataset that starts with source prefix, e.g s3://"
            )
        default_version = 1
        try:
            dataset = self.get_dataset(name)
            default_version = dataset.next_version
        except DatasetNotFoundError:
            schema = {
                c.name: c.type.to_dict() for c in columns if isinstance(c.type, SQLType)
            }
            dataset = self.metastore.create_dataset(
                name,
                feature_schema=feature_schema,
                query_script=query_script,
                schema=schema,
                ignore_if_exists=True,
            )

        version = version or default_version

        if dataset.has_version(version):
            raise DatasetInvalidVersionError(
                f"Version {version} already exists in dataset {name}"
            )

        if validate_version and not dataset.is_valid_next_version(version):
            raise DatasetInvalidVersionError(
                f"Version {version} must be higher than the current latest one"
            )

        return self.create_new_dataset_version(
            dataset,
            version,
            feature_schema=feature_schema,
            query_script=query_script,
            create_rows_table=create_rows,
            columns=columns,
        )

    def create_new_dataset_version(
        self,
        dataset: DatasetRecord,
        version: int,
        *,
        columns: Sequence[Column],
        sources="",
        feature_schema=None,
        query_script="",
        error_message="",
        error_stack="",
        script_output="",
        create_rows_table=True,
        job_id: Optional[str] = None,
        is_job_result: bool = False,
    ) -> DatasetRecord:
        """
        Creates dataset version if it doesn't exist.
        If create_rows is False, dataset rows table will not be created
        """
        assert [c.name for c in columns if c.name != "sys__id"], f"got {columns=}"
        schema = {
            c.name: c.type.to_dict() for c in columns if isinstance(c.type, SQLType)
        }
        dataset = self.metastore.create_dataset_version(
            dataset,
            version,
            status=DatasetStatus.PENDING,
            sources=sources,
            feature_schema=feature_schema,
            query_script=query_script,
            error_message=error_message,
            error_stack=error_stack,
            script_output=script_output,
            schema=schema,
            job_id=job_id,
            is_job_result=is_job_result,
            ignore_if_exists=True,
        )

        if create_rows_table:
            table_name = self.warehouse.dataset_table_name(dataset.name, version)
            self.warehouse.create_dataset_rows_table(table_name, columns=columns)
            self.update_dataset_version_with_warehouse_info(dataset, version)

        return dataset

    def update_dataset_version_with_warehouse_info(
        self, dataset: DatasetRecord, version: int, rows_dropped=False, **kwargs
    ) -> None:
        from datachain.query import DatasetQuery

        dataset_version = dataset.get_version(version)

        values = {**kwargs}

        if rows_dropped:
            values["num_objects"] = None
            values["size"] = None
            values["preview"] = None
            self.metastore.update_dataset_version(
                dataset,
                version,
                **values,
            )
            return

        if not dataset_version.num_objects:
            num_objects, size = self.warehouse.dataset_stats(dataset, version)
            if num_objects != dataset_version.num_objects:
                values["num_objects"] = num_objects
            if size != dataset_version.size:
                values["size"] = size

        if not dataset_version.preview:
            values["preview"] = (
                DatasetQuery(name=dataset.name, version=version, catalog=self)
                .limit(20)
                .to_db_records()
            )

        if not values:
            return

        self.metastore.update_dataset_version(
            dataset,
            version,
            **values,
        )

    def update_dataset(
        self, dataset: DatasetRecord, conn=None, **kwargs
    ) -> DatasetRecord:
        """Updates dataset fields."""
        old_name = None
        new_name = None
        if "name" in kwargs and kwargs["name"] != dataset.name:
            old_name = dataset.name
            new_name = kwargs["name"]

        dataset = self.metastore.update_dataset(dataset, conn=conn, **kwargs)

        if old_name and new_name:
            # updating name must result in updating dataset table names as well
            for version in [v.version for v in dataset.versions]:
                self.warehouse.rename_dataset_table(
                    old_name,
                    new_name,
                    old_version=version,
                    new_version=version,
                )

        return dataset

    def remove_dataset_version(
        self, dataset: DatasetRecord, version: int, drop_rows: Optional[bool] = True
    ) -> None:
        """
        Deletes one single dataset version.
        If it was last version, it removes dataset completely
        """
        if not dataset.has_version(version):
            return
        dataset = self.metastore.remove_dataset_version(dataset, version)
        if drop_rows:
            self.warehouse.drop_dataset_rows_table(dataset, version)

    def get_temp_table_names(self) -> list[str]:
        return self.warehouse.get_temp_table_names()

    def cleanup_tables(self, names: Iterable[str]) -> None:
        """
        Drop tables passed.

        This should be implemented to ensure that the provided tables
        are cleaned up as soon as they are no longer needed.
        """
        self.warehouse.cleanup_tables(names)
        self.id_generator.delete_uris(names)

    def create_dataset_from_sources(
        self,
        name: str,
        sources: list[str],
        client_config=None,
        recursive=False,
    ) -> DatasetRecord:
        if not sources:
            raise ValueError("Sources needs to be non empty list")

        from datachain.query import DatasetQuery

        dataset_queries = []
        for source in sources:
            if source.startswith(DATASET_PREFIX):
                dq = DatasetQuery(
                    name=source[len(DATASET_PREFIX) :],
                    catalog=self,
                    client_config=client_config,
                )
            else:
                dq = DatasetQuery(
                    path=source,
                    catalog=self,
                    client_config=client_config,
                    recursive=recursive,
                )

            dataset_queries.append(dq)

        # create union of all dataset queries created from sources
        dq = reduce(lambda ds1, ds2: ds1.union(ds2), dataset_queries)
        try:
            dq.save(name)
        except Exception as e:  # noqa: BLE001
            try:
                ds = self.get_dataset(name)
                self.metastore.update_dataset_status(
                    ds,
                    DatasetStatus.FAILED,
                    version=ds.latest_version,
                    error_message=DATASET_INTERNAL_ERROR_MESSAGE,
                    error_stack=traceback.format_exc(),
                )
                self._remove_dataset_rows_and_warehouse_info(
                    ds,
                    ds.latest_version,
                    sources="\n".join(sources),
                )
                raise
            except DatasetNotFoundError:
                raise e from None

        ds = self.get_dataset(name)

        self.update_dataset_version_with_warehouse_info(
            ds,
            ds.latest_version,
            sources="\n".join(sources),
        )

        return self.get_dataset(name)

    def register_new_dataset(
        self,
        source_dataset: DatasetRecord,
        source_version: int,
        target_name: str,
    ) -> DatasetRecord:
        target_dataset = self.metastore.create_dataset(
            target_name,
            query_script=source_dataset.query_script,
            schema=source_dataset.serialized_schema,
        )
        return self.register_dataset(source_dataset, source_version, target_dataset, 1)

    def register_dataset(
        self,
        dataset: DatasetRecord,
        version: int,
        target_dataset: DatasetRecord,
        target_version: Optional[int] = None,
    ) -> DatasetRecord:
        """
        Registers dataset version of one dataset as dataset version of another
        one (it can be new version of existing one).
        It also removes original dataset version
        """
        target_version = target_version or target_dataset.next_version

        if not target_dataset.is_valid_next_version(target_version):
            raise DatasetInvalidVersionError(
                f"Version {target_version} must be higher than the current latest one"
            )

        dataset_version = dataset.get_version(version)
        if not dataset_version:
            raise ValueError(f"Dataset {dataset.name} does not have version {version}")

        if not dataset_version.is_final_status():
            raise ValueError("Cannot register dataset version in non final status")

        # copy dataset version
        target_dataset = self.metastore.create_dataset_version(
            target_dataset,
            target_version,
            sources=dataset_version.sources,
            status=dataset_version.status,
            query_script=dataset_version.query_script,
            error_message=dataset_version.error_message,
            error_stack=dataset_version.error_stack,
            script_output=dataset_version.script_output,
            created_at=dataset_version.created_at,
            finished_at=dataset_version.finished_at,
            schema=dataset_version.serialized_schema,
            num_objects=dataset_version.num_objects,
            size=dataset_version.size,
            preview=dataset_version.preview,
            job_id=dataset_version.job_id,
            is_job_result=dataset_version.is_job_result,
        )
        # to avoid re-creating rows table, we are just renaming it for a new version
        # of target dataset
        self.warehouse.rename_dataset_table(
            dataset.name,
            target_dataset.name,
            old_version=version,
            new_version=target_version,
        )
        self.metastore.update_dataset_dependency_source(
            dataset,
            version,
            new_source_dataset=target_dataset,
            new_source_dataset_version=target_version,
        )

        if dataset.id == target_dataset.id:
            # we are updating the same dataset so we need to refresh it to have newly
            # added version in step before
            dataset = self.get_dataset(dataset.name)

        self.remove_dataset_version(dataset, version, drop_rows=False)

        return self.get_dataset(target_dataset.name)

    def get_dataset(self, name: str) -> DatasetRecord:
        return self.metastore.get_dataset(name)

    def get_remote_dataset(self, name: str, *, remote_config=None) -> DatasetRecord:
        remote_config = remote_config or get_remote_config(
            read_config(DataChainDir.find().root), remote=""
        )
        studio_client = StudioClient(
            remote_config["url"], remote_config["username"], remote_config["token"]
        )

        info_response = studio_client.dataset_info(name)
        if not info_response.ok:
            _raise_remote_error(info_response.message)

        dataset_info = info_response.data
        assert isinstance(dataset_info, dict)
        return DatasetRecord.from_dict(dataset_info)

    def get_dataset_dependencies(
        self, name: str, version: int, indirect=False
    ) -> list[Optional[DatasetDependency]]:
        dataset = self.get_dataset(name)

        direct_dependencies = self.metastore.get_direct_dataset_dependencies(
            dataset, version
        )

        if not indirect:
            return direct_dependencies

        for d in direct_dependencies:
            if not d:
                # dependency has been removed
                continue
            if d.is_dataset:
                # only datasets can have dependencies
                d.dependencies = self.get_dataset_dependencies(
                    d.name, int(d.version), indirect=indirect
                )

        return direct_dependencies

    def ls_datasets(self) -> Iterator[DatasetRecord]:
        datasets = self.metastore.list_datasets()
        for d in datasets:
            if not d.is_bucket_listing:
                yield d

    def list_datasets_versions(
        self,
    ) -> Iterator[tuple[DatasetRecord, "DatasetVersion", Optional["Job"]]]:
        """Iterate over all dataset versions with related jobs."""
        datasets = list(self.ls_datasets())

        # preselect dataset versions jobs from db to avoid multiple queries
        jobs_ids: set[str] = {
            v.job_id for ds in datasets for v in ds.versions if v.job_id
        }
        jobs: dict[str, Job] = {}
        if jobs_ids:
            jobs = {j.id: j for j in self.metastore.list_jobs_by_ids(list(jobs_ids))}

        for d in datasets:
            yield from (
                (d, v, jobs.get(v.job_id) if v.job_id else None) for v in d.versions
            )

    def ls_dataset_rows(
        self, name: str, version: int, offset=None, limit=None
    ) -> list[dict]:
        from datachain.query import DatasetQuery

        dataset = self.get_dataset(name)

        q = DatasetQuery(name=dataset.name, version=version, catalog=self)
        if limit:
            q = q.limit(limit)
        if offset:
            q = q.offset(offset)

        q = q.order_by("sys__id")

        return q.to_db_records()

    def signed_url(self, source: str, path: str, client_config=None) -> str:
        client_config = client_config or self.client_config
        client, _ = self.parse_url(source, **client_config)
        return client.url(path)

    def export_dataset_table(
        self,
        bucket_uri: str,
        name: str,
        version: int,
        client_config=None,
    ) -> list[str]:
        dataset = self.get_dataset(name)

        return self.warehouse.export_dataset_table(
            bucket_uri, dataset, version, client_config
        )

    def dataset_table_export_file_names(self, name: str, version: int) -> list[str]:
        dataset = self.get_dataset(name)
        return self.warehouse.dataset_table_export_file_names(dataset, version)

    def dataset_stats(self, name: str, version: int) -> DatasetStats:
        """
        Returns tuple with dataset stats: total number of rows and total dataset size.
        """
        dataset = self.get_dataset(name)
        dataset_version = dataset.get_version(version)
        return DatasetStats(
            num_objects=dataset_version.num_objects,
            size=dataset_version.size,
        )

    def remove_dataset(
        self,
        name: str,
        version: Optional[int] = None,
        force: Optional[bool] = False,
    ):
        dataset = self.get_dataset(name)
        if not version and not force:
            raise ValueError(f"Missing dataset version from input for dataset {name}")
        if version and not dataset.has_version(version):
            raise DatasetInvalidVersionError(
                f"Dataset {name} doesn't have version {version}"
            )

        if version:
            self.remove_dataset_version(dataset, version)
            return

        while dataset.versions:
            version = dataset.versions[0].version
            self.remove_dataset_version(
                dataset,
                version,
            )

    def edit_dataset(
        self,
        name: str,
        new_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[list[str]] = None,
    ) -> DatasetRecord:
        update_data = {}
        if new_name:
            update_data["name"] = new_name
        if description is not None:
            update_data["description"] = description
        if labels is not None:
            update_data["labels"] = labels  # type: ignore[assignment]

        dataset = self.get_dataset(name)
        return self.update_dataset(dataset, **update_data)

    def get_file_signals(
        self, dataset_name: str, dataset_version: int, row: RowDict
    ) -> Optional[dict]:
        """
        Function that returns file signals from dataset row.
        Note that signal names are without prefix, so if there was 'laion__file__source'
        in original row, result will have just 'source'
        Example output:
            {
                "source": "s3://ldb-public",
                "path": "animals/dogs/dog.jpg",
                ...
            }
        """
        from datachain.lib.file import File
        from datachain.lib.signal_schema import DEFAULT_DELIMITER, SignalSchema

        version = self.get_dataset(dataset_name).get_version(dataset_version)

        file_signals_values = {}

        schema = SignalSchema.deserialize(version.feature_schema)
        for file_signals in schema.get_signals(File):
            prefix = file_signals.replace(".", DEFAULT_DELIMITER) + DEFAULT_DELIMITER
            file_signals_values[file_signals] = {
                c_name.removeprefix(prefix): c_value
                for c_name, c_value in row.items()
                if c_name.startswith(prefix)
                and DEFAULT_DELIMITER not in c_name.removeprefix(prefix)
            }

        if not file_signals_values:
            return None

        # there can be multiple file signals in a schema, but taking the first
        # one for now. In future we might add ability to choose from which one
        # to open object
        return next(iter(file_signals_values.values()))

    def open_object(
        self,
        dataset_name: str,
        dataset_version: int,
        row: RowDict,
        use_cache: bool = True,
        **config: Any,
    ):
        file_signals = self.get_file_signals(dataset_name, dataset_version, row)
        if not file_signals:
            raise RuntimeError("Cannot open object without file signals")

        config = config or self.client_config
        client = self.get_client(file_signals["source"], **config)
        return client.open_object(
            self._get_row_uid(file_signals),  # type: ignore [arg-type]
            use_cache=use_cache,
        )

    def _get_row_uid(self, row: RowDict) -> UniqueId:
        return UniqueId(
            row["source"],
            row["path"],
            row["size"],
            row["etag"],
            row["version"],
            row["is_latest"],
            row["vtype"],
            row["location"],
            row["last_modified"],
        )

    def ls(
        self,
        sources: list[str],
        fields: Iterable[str],
        ttl=TTL_INT,
        update=False,
        skip_indexing=False,
        *,
        client_config=None,
    ) -> Iterator[tuple[DataSource, Iterable[tuple]]]:
        data_sources = self.enlist_sources(
            sources,
            ttl,
            update,
            skip_indexing=skip_indexing,
            client_config=client_config or self.client_config,
        )

        for source in data_sources:  # type: ignore [union-attr]
            yield source, source.ls(fields)

    def ls_storage_uris(self) -> Iterator[str]:
        yield from self.metastore.get_all_storage_uris()

    def get_storage(self, uri: StorageURI) -> Storage:
        return self.metastore.get_storage(uri)

    def ls_storages(self) -> list[Storage]:
        return self.metastore.list_storages()

    def pull_dataset(
        self,
        dataset_uri: str,
        output: Optional[str] = None,
        no_cp: bool = False,
        force: bool = False,
        edatachain: bool = False,
        edatachain_file: Optional[str] = None,
        *,
        client_config=None,
        remote_config=None,
    ) -> None:
        # TODO add progress bar https://github.com/iterative/dvcx/issues/750
        # TODO copy correct remote dates https://github.com/iterative/dvcx/issues/new
        # TODO compare dataset stats on remote vs local pull to assert it's ok
        def _instantiate_dataset():
            if no_cp:
                return
            self.cp(
                [dataset_uri],
                output,
                force=force,
                no_edatachain_file=not edatachain,
                edatachain_file=edatachain_file,
                client_config=client_config,
            )
            print(f"Dataset {dataset_uri} instantiated locally to {output}")

        if not output and not no_cp:
            raise ValueError("Please provide output directory for instantiation")

        client_config = client_config or self.client_config
        remote_config = remote_config or get_remote_config(
            read_config(DataChainDir.find().root), remote=""
        )

        studio_client = StudioClient(
            remote_config["url"], remote_config["username"], remote_config["token"]
        )

        try:
            remote_dataset_name, version = parse_dataset_uri(dataset_uri)
        except Exception as e:
            raise DataChainError("Error when parsing dataset uri") from e

        dataset = None
        try:
            dataset = self.get_dataset(remote_dataset_name)
        except DatasetNotFoundError:
            # we will create new one if it doesn't exist
            pass

        remote_dataset = self.get_remote_dataset(
            remote_dataset_name, remote_config=remote_config
        )
        # if version is not specified in uri, take the latest one
        if not version:
            version = remote_dataset.latest_version
            print(f"Version not specified, pulling the latest one (v{version})")
            # updating dataset uri with latest version
            dataset_uri = create_dataset_uri(remote_dataset_name, version)

        assert version

        if dataset and dataset.has_version(version):
            print(f"Local copy of dataset {dataset_uri} already present")
            _instantiate_dataset()
            return

        try:
            remote_dataset_version = remote_dataset.get_version(version)
        except (ValueError, StopIteration) as exc:
            raise DataChainError(
                f"Dataset {remote_dataset_name} doesn't have version {version}"
                " on server"
            ) from exc

        stats_response = studio_client.dataset_stats(remote_dataset_name, version)
        if not stats_response.ok:
            _raise_remote_error(stats_response.message)
        dataset_stats = stats_response.data

        dataset_save_progress_bar = tqdm(
            desc=f"Saving dataset {dataset_uri} locally: ",
            unit=" rows",
            unit_scale=True,
            unit_divisor=1000,
            total=dataset_stats.num_objects,  # type: ignore [union-attr]
        )

        schema = DatasetRecord.parse_schema(remote_dataset_version.schema)

        columns = tuple(
            sa.Column(name, typ) for name, typ in schema.items() if name != "sys__id"
        )
        # creating new dataset (version) locally
        dataset = self.create_dataset(
            remote_dataset_name,
            version,
            query_script=remote_dataset_version.query_script,
            create_rows=True,
            columns=columns,
            validate_version=False,
        )

        # asking remote to export dataset rows table to s3 and to return signed
        # urls of exported parts, which are in parquet format
        export_response = studio_client.export_dataset_table(
            remote_dataset_name, version
        )
        if not export_response.ok:
            _raise_remote_error(export_response.message)

        signed_urls = export_response.data

        if signed_urls:
            shuffle(signed_urls)

            with (
                self.metastore.clone() as metastore,
                self.warehouse.clone() as warehouse,
            ):
                rows_fetcher = DatasetRowsFetcher(
                    metastore,
                    warehouse,
                    remote_config,
                    dataset.name,
                    version,
                    schema,
                )
                try:
                    rows_fetcher.run(
                        batched(
                            signed_urls,
                            math.ceil(len(signed_urls) / PULL_DATASET_MAX_THREADS),
                        ),
                        dataset_save_progress_bar,
                    )
                except:
                    self.remove_dataset(dataset.name, version)
                    raise

        dataset = self.metastore.update_dataset_status(
            dataset,
            DatasetStatus.COMPLETE,
            version=version,
            error_message=remote_dataset.error_message,
            error_stack=remote_dataset.error_stack,
            script_output=remote_dataset.error_stack,
        )
        self.update_dataset_version_with_warehouse_info(dataset, version)

        dataset_save_progress_bar.close()
        print(f"Dataset {dataset_uri} saved locally")

        _instantiate_dataset()

    def clone(
        self,
        sources: list[str],
        output: str,
        force: bool = False,
        update: bool = False,
        recursive: bool = False,
        no_glob: bool = False,
        no_cp: bool = False,
        edatachain: bool = False,
        edatachain_file: Optional[str] = None,
        ttl: int = TTL_INT,
        *,
        client_config=None,
    ) -> None:
        """
        This command takes cloud path(s) and duplicates files and folders in
        them into the dataset folder.
        It also adds those files to a dataset in database, which is
        created if doesn't exist yet
        Optionally, it creates a .edatachain file
        """
        if not no_cp or edatachain:
            self.cp(
                sources,
                output,
                force=force,
                update=update,
                recursive=recursive,
                no_glob=no_glob,
                edatachain_only=no_cp,
                no_edatachain_file=not edatachain,
                edatachain_file=edatachain_file,
                ttl=ttl,
                client_config=client_config,
            )
        else:
            # since we don't call cp command, which does listing implicitly,
            # it needs to be done here
            self.enlist_sources(
                sources,
                ttl,
                update,
                client_config=client_config or self.client_config,
            )

        self.create_dataset_from_sources(
            output, sources, client_config=client_config, recursive=recursive
        )

    def apply_udf(
        self,
        udf_location: str,
        source: str,
        target_name: str,
        parallel: Optional[int] = None,
        params: Optional[str] = None,
    ):
        from datachain.query import DatasetQuery

        if source.startswith(DATASET_PREFIX):
            ds = DatasetQuery(name=source[len(DATASET_PREFIX) :], catalog=self)
        else:
            ds = DatasetQuery(path=source, catalog=self)
        udf = import_object(udf_location)
        if params:
            args, kwargs = parse_params_string(params)
            udf = udf(*args, **kwargs)
        ds.add_signals(udf, parallel=parallel).save(target_name)

    def query(
        self,
        query_script: str,
        envs: Optional[Mapping[str, str]] = None,
        python_executable: Optional[str] = None,
        save: bool = False,
        save_as: Optional[str] = None,
        preview_limit: int = 10,
        preview_offset: int = 0,
        preview_columns: Optional[list[str]] = None,
        capture_output: bool = True,
        output_hook: Callable[[str], None] = noop,
        params: Optional[dict[str, str]] = None,
        job_id: Optional[str] = None,
    ) -> QueryResult:
        """
        Method to run custom user Python script to run a query and, as result,
        creates new dataset from the results of a query.
        Returns tuple of result dataset and script output.

        Constraints on query script:
            1. datachain.query.DatasetQuery should be used in order to create query
            for a dataset
            2. There should not be any .save() call on DatasetQuery since the idea
            is to create only one dataset as the outcome of the script
            3. Last statement must be an instance of DatasetQuery

        If save is set to True, we are creating new dataset with results
        from dataset query. If it's set to False, we will just print results
        without saving anything

        Example of query script:
            from datachain.query import DatasetQuery, C
            DatasetQuery('s3://ldb-public/remote/datasets/mnist-tiny/').filter(
                C.size > 1000
            )
        """
        from datachain.query.dataset import ExecutionResult

        feature_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            dir=os.getcwd(), suffix=".py", delete=False
        )
        _, feature_module = os.path.split(feature_file.name)

        try:
            lines, proc, response_text = self.run_query(
                python_executable or sys.executable,
                query_script,
                envs,
                feature_file,
                capture_output,
                feature_module,
                output_hook,
                params,
                preview_columns,
                preview_limit,
                preview_offset,
                save,
                save_as,
                job_id,
            )
        finally:
            feature_file.close()
            os.unlink(feature_file.name)

        output = "".join(lines)

        if proc.returncode:
            if proc.returncode == QUERY_SCRIPT_CANCELED_EXIT_CODE:
                raise QueryScriptCancelError(
                    "Query script was canceled by user",
                    return_code=proc.returncode,
                    output=output,
                )
            if proc.returncode == QUERY_SCRIPT_INVALID_LAST_STATEMENT_EXIT_CODE:
                raise QueryScriptRunError(
                    "Last line in a script was not an instance of DataChain",
                    return_code=proc.returncode,
                    output=output,
                )
            raise QueryScriptRunError(
                f"Query script exited with error code {proc.returncode}",
                return_code=proc.returncode,
                output=output,
            )

        try:
            response = json.loads(response_text)
        except ValueError:
            response = {}
        exec_result = ExecutionResult(**response)

        dataset: Optional[DatasetRecord] = None
        version: Optional[int] = None
        if save or save_as:
            dataset, version = self.save_result(
                query_script, exec_result, output, version, job_id
            )

        return QueryResult(
            dataset=dataset,
            version=version,
            output=output,
            preview=exec_result.preview,
            metrics=exec_result.metrics,
        )

    def run_query(
        self,
        python_executable: str,
        query_script: str,
        envs: Optional[Mapping[str, str]],
        feature_file: IO[bytes],
        capture_output: bool,
        feature_module: str,
        output_hook: Callable[[str], None],
        params: Optional[dict[str, str]],
        preview_columns: Optional[list[str]],
        preview_limit: int,
        preview_offset: int,
        save: bool,
        save_as: Optional[str],
        job_id: Optional[str],
    ) -> tuple[list[str], subprocess.Popen, str]:
        try:
            feature_code, query_script_compiled = self.compile_query_script(
                query_script, feature_module[:-3]
            )
            if feature_code:
                feature_file.write(feature_code.encode())
                feature_file.flush()

        except Exception as exc:
            raise QueryScriptCompileError(
                f"Query script failed to compile, reason: {exc}"
            ) from exc
        if save_as and save_as.startswith(QUERY_DATASET_PREFIX):
            raise ValueError(
                f"Cannot use {QUERY_DATASET_PREFIX} prefix for dataset name"
            )
        r, w = os.pipe()
        if os.name == "nt":
            import msvcrt

            os.set_inheritable(w, True)

            startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
            handle = msvcrt.get_osfhandle(w)  # type: ignore[attr-defined]
            startupinfo.lpAttributeList["handle_list"].append(handle)
            kwargs: dict[str, Any] = {"startupinfo": startupinfo}
        else:
            handle = w
            kwargs = {"pass_fds": [w]}
        envs = dict(envs or os.environ)
        if feature_code:
            envs["DATACHAIN_FEATURE_CLASS_SOURCE"] = json.dumps(
                {feature_module: feature_code}
            )
        envs.update(
            {
                "DATACHAIN_QUERY_PARAMS": json.dumps(params or {}),
                "PYTHONPATH": os.getcwd(),  # For local imports
                "DATACHAIN_QUERY_PREVIEW_ARGS": json.dumps(
                    {
                        "limit": preview_limit,
                        "offset": preview_offset,
                        "columns": preview_columns,
                    }
                ),
                "DATACHAIN_QUERY_SAVE": "1" if save else "",
                "DATACHAIN_QUERY_SAVE_AS": save_as or "",
                "PYTHONUNBUFFERED": "1",
                "DATACHAIN_OUTPUT_FD": str(handle),
                "DATACHAIN_JOB_ID": job_id or "",
            },
        )
        with subprocess.Popen(  # noqa: S603
            [python_executable, "-c", query_script_compiled],
            env=envs,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.STDOUT if capture_output else None,
            bufsize=1,
            text=False,
            **kwargs,
        ) as proc:
            os.close(w)

            out = proc.stdout
            _lines: list[str] = []
            ctx = print_and_capture(out, output_hook) if out else nullcontext(_lines)

            with ctx as lines, open(r) as f:
                response_text = ""
                while proc.poll() is None:
                    response_text += f.readline()
                    time.sleep(0.1)
                response_text += f.readline()
        return lines, proc, response_text

    def save_result(self, query_script, exec_result, output, version, job_id):
        if not exec_result.dataset:
            raise QueryScriptDatasetNotFound(
                "No dataset found after running Query script",
                output=output,
            )
        name, version = exec_result.dataset
        # finding returning dataset
        try:
            dataset = self.get_dataset(name)
            dataset.get_version(version)
        except (DatasetNotFoundError, ValueError) as e:
            raise QueryScriptDatasetNotFound(
                "No dataset found after running Query script",
                output=output,
            ) from e
        dataset = self.update_dataset(
            dataset,
            script_output=output,
            query_script=query_script,
        )
        self.update_dataset_version_with_warehouse_info(
            dataset,
            version,
            script_output=output,
            query_script=query_script,
            job_id=job_id,
            is_job_result=True,
        )
        return dataset, version

    def cp(
        self,
        sources: list[str],
        output: str,
        force: bool = False,
        update: bool = False,
        recursive: bool = False,
        edatachain_file: Optional[str] = None,
        edatachain_only: bool = False,
        no_edatachain_file: bool = False,
        no_glob: bool = False,
        ttl: int = TTL_INT,
        *,
        client_config=None,
    ) -> list[dict[str, Any]]:
        """
        This function copies files from cloud sources to local destination directory
        If cloud source is not indexed, or has expired index, it runs indexing
        It also creates .edatachain file by default, if not specified differently
        """
        client_config = client_config or self.client_config
        node_groups = self.enlist_sources_grouped(
            sources,
            ttl,
            update,
            no_glob,
            client_config=client_config,
        )

        always_copy_dir_contents, copy_to_filename = prepare_output_for_cp(
            node_groups, output, force, edatachain_only, no_edatachain_file
        )
        dataset_file = check_output_dataset_file(
            output, force, edatachain_file, no_edatachain_file
        )

        total_size, total_files = collect_nodes_for_cp(node_groups, recursive)

        if total_files == 0:
            # Nothing selected to cp
            return []

        desc_max_len = max(len(output) + 16, 19)
        bar_format = (
            "{desc:<"
            f"{desc_max_len}"
            "}{percentage:3.0f}%|{bar}| {n_fmt:>5}/{total_fmt:<5} "
            "[{elapsed}<{remaining}, {rate_fmt:>8}]"
        )

        if not edatachain_only:
            with get_download_bar(bar_format, total_size) as pbar:
                for node_group in node_groups:
                    node_group.download(recursive=recursive, pbar=pbar)

        instantiate_node_groups(
            node_groups,
            output,
            bar_format,
            total_files,
            force,
            recursive,
            edatachain_only,
            always_copy_dir_contents,
            copy_to_filename,
        )
        if no_edatachain_file:
            return []

        metafile_data = compute_metafile_data(node_groups)
        if metafile_data:
            # Don't write the metafile if nothing was copied
            print(f"Creating '{dataset_file}'")
            with open(dataset_file, "w", encoding="utf-8") as fd:
                yaml.dump(metafile_data, fd, sort_keys=False)

        return metafile_data

    def du(
        self,
        sources,
        depth=0,
        ttl=TTL_INT,
        update=False,
        *,
        client_config=None,
    ) -> Iterable[tuple[str, float]]:
        sources = self.enlist_sources(
            sources,
            ttl,
            update,
            client_config=client_config or self.client_config,
        )

        def du_dirs(src, node, subdepth):
            if subdepth > 0:
                subdirs = src.listing.get_dirs_by_parent_path(node.path)
                for sd in subdirs:
                    yield from du_dirs(src, sd, subdepth - 1)
            yield (
                src.get_node_full_path(node),
                src.listing.du(node)[0],
            )

        for src in sources:
            yield from du_dirs(src, src.node, depth)

    def find(
        self,
        sources,
        ttl=TTL_INT,
        update=False,
        names=None,
        inames=None,
        paths=None,
        ipaths=None,
        size=None,
        typ=None,
        columns=None,
        *,
        client_config=None,
    ) -> Iterator[str]:
        sources = self.enlist_sources(
            sources,
            ttl,
            update,
            client_config=client_config or self.client_config,
        )
        if not columns:
            columns = ["path"]
        field_set = set()
        for column in columns:
            if column == "du":
                field_set.add("dir_type")
                field_set.add("size")
                field_set.add("path")
            elif column == "name":
                field_set.add("path")
            elif column == "owner":
                field_set.add("owner_name")
            elif column == "path":
                field_set.add("dir_type")
                field_set.add("path")
            elif column == "size":
                field_set.add("size")
            elif column == "type":
                field_set.add("dir_type")
        fields = list(field_set)
        field_lookup = {f: i for i, f in enumerate(fields)}
        for src in sources:
            results = src.listing.find(
                src.node, fields, names, inames, paths, ipaths, size, typ
            )
            for row in results:
                yield "\t".join(
                    find_column_to_str(row, field_lookup, src, column)
                    for column in columns
                )

    def index(
        self,
        sources,
        ttl=TTL_INT,
        update=False,
        *,
        client_config=None,
    ) -> None:
        root_sources = [
            src for src in sources if Client.get_implementation(src).is_root_url(src)
        ]
        non_root_sources = [
            src
            for src in sources
            if not Client.get_implementation(src).is_root_url(src)
        ]

        client_config = client_config or self.client_config

        # for root sources (e.g s3://) we are just getting all buckets and
        # saving them as storages, without further indexing in each bucket
        for source in root_sources:
            for bucket in Client.get_implementation(source).ls_buckets(**client_config):
                client = self.get_client(bucket.uri, **client_config)
                print(f"Registering storage {client.uri}")
                self.metastore.create_storage_if_not_registered(client.uri)

        self.enlist_sources(
            non_root_sources,
            ttl,
            update,
            client_config=client_config,
            only_index=True,
        )

    def find_stale_storages(self) -> None:
        self.metastore.find_stale_storages()
