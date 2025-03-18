import glob
import io
import json
import logging
import os
import os.path as osp
import random
import re
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timezone
from itertools import chain, islice
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union
from uuid import UUID

import cloudpickle
import platformdirs
from dateutil import tz
from dateutil.parser import isoparse
from pydantic import BaseModel

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self


logger = logging.getLogger("datachain")

NUL = b"\0"
TIME_ZERO = datetime.fromtimestamp(0, tz=timezone.utc)

APPNAME = "datachain"
APPAUTHOR = "iterative"
ENV_DATACHAIN_SYSTEM_CONFIG_DIR = "DATACHAIN_SYSTEM_CONFIG_DIR"
ENV_DATACHAIN_GLOBAL_CONFIG_DIR = "DATACHAIN_GLOBAL_CONFIG_DIR"
STUDIO_URL = "https://studio.datachain.ai"


T = TypeVar("T")


class DataChainDir:
    DEFAULT = ".datachain"
    CACHE = "cache"
    TMP = "tmp"
    DB = "db"
    CONFIG = "config"
    ENV_VAR = "DATACHAIN_DIR"
    ENV_VAR_DATACHAIN_ROOT = "DATACHAIN_ROOT_DIR"

    def __init__(
        self,
        root: Optional[str] = None,
        cache: Optional[str] = None,
        tmp: Optional[str] = None,
        db: Optional[str] = None,
        config: Optional[str] = None,
    ) -> None:
        self.root = osp.abspath(root) if root is not None else self.default_root()
        self.cache = (
            osp.abspath(cache) if cache is not None else osp.join(self.root, self.CACHE)
        )
        self.tmp = (
            osp.abspath(tmp) if tmp is not None else osp.join(self.root, self.TMP)
        )
        self.db = osp.abspath(db) if db is not None else osp.join(self.root, self.DB)
        self.config = (
            osp.abspath(config)
            if config is not None
            else osp.join(self.root, self.CONFIG)
        )
        self.config = (
            osp.abspath(config)
            if config is not None
            else osp.join(self.root, self.CONFIG)
        )

    def init(self):
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.cache, exist_ok=True)
        os.makedirs(self.tmp, exist_ok=True)
        os.makedirs(osp.split(self.db)[0], exist_ok=True)
        os.makedirs(osp.split(self.config)[0], exist_ok=True)
        os.makedirs(osp.split(self.config)[0], exist_ok=True)

    @classmethod
    def default_root(cls) -> str:
        try:
            root_dir = os.environ[cls.ENV_VAR_DATACHAIN_ROOT]
        except KeyError:
            root_dir = os.getcwd()

        return osp.join(root_dir, cls.DEFAULT)

    @classmethod
    def find(cls, create: bool = True) -> "Self":
        try:
            root = os.environ[cls.ENV_VAR]
        except KeyError:
            root = cls.default_root()
        instance = cls(root)
        if not osp.isdir(root):
            if create:
                instance.init()
            else:
                raise NotADirectoryError(root)
        return instance


def system_config_dir():
    return os.getenv(ENV_DATACHAIN_SYSTEM_CONFIG_DIR) or platformdirs.site_config_dir(
        APPNAME, APPAUTHOR
    )


def global_config_dir():
    return os.getenv(ENV_DATACHAIN_GLOBAL_CONFIG_DIR) or platformdirs.user_config_dir(
        APPNAME, APPAUTHOR
    )


def human_time_to_int(time: str) -> Optional[int]:
    if not time:
        return None

    suffix = time[-1]
    try:
        num = int(time if suffix.isdigit() else time[:-1])
    except ValueError:
        return None
    return num * {
        "h": 60 * 60,
        "d": 60 * 60 * 24,
        "w": 60 * 60 * 24 * 7,
        "m": 31 * 24 * 60 * 60,
        "y": 60 * 60 * 24 * 365,
    }.get(suffix.lower(), 1)


def time_to_str(dt):
    if isinstance(dt, str):
        dt = isoparse(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def time_to_local(dt: Union[datetime, str]) -> datetime:
    # TODO check usage
    if isinstance(dt, str):
        dt = isoparse(dt)
    try:
        return dt.astimezone(tz.tzlocal())
    except (OverflowError, OSError, ValueError):
        return dt


def time_to_local_str(dt: Union[datetime, str]) -> str:
    return time_to_str(time_to_local(dt))


def is_expired(expires: Optional[Union[datetime, str]]):
    if expires:
        return time_to_local(expires) < time_to_local(datetime.now())  # noqa: DTZ005

    return False


SIZE_SUFFIXES = ["", "K", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"]


def sizeof_fmt(num, suffix="", si=False):
    power = 1000.0 if si else 1024.0
    for unit in SIZE_SUFFIXES[:-1]:
        if abs(num) < power:
            if not unit:
                return f"{num:4.0f}{suffix}"
            return f"{num:3.1f}{unit}{suffix}"
        num /= power
    return f"{num:.1f}Q{suffix}"


def suffix_to_number(num_str: str) -> int:
    try:
        if len(num_str) > 1:
            suffix = num_str[-1].upper()
            if suffix in SIZE_SUFFIXES:
                suffix_idx = SIZE_SUFFIXES.index(suffix)
                return int(num_str[:-1]) * (1024**suffix_idx)
        return int(num_str)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid number/suffix for: {num_str}") from None


def datachain_paths_join(source_path: str, file_paths: Iterable[str]) -> Iterable[str]:
    source_parts = source_path.rstrip("/").split("/")
    if glob.has_magic(source_parts[-1]):
        # Remove last element if it is a glob match (such as *)
        source_parts.pop()
    source_stripped = "/".join(source_parts)
    return (f"{source_stripped}/{path.lstrip('/')}" for path in file_paths)


def sql_escape_like(search: str, escape: str = "\\") -> str:
    return (
        search.replace(escape, escape * 2)
        .replace("%", f"{escape}%")
        .replace("_", f"{escape}_")
    )


def get_envs_by_prefix(prefix: str) -> dict[str, str]:
    """
    Function that searches env variables by some name prefix and returns
    the ones found, but with prefix being excluded from it's names
    """
    variables: dict[str, str] = {}
    for env_name, env_value in os.environ.items():
        if env_name.startswith(prefix):
            variables[env_name[len(prefix) :]] = env_value

    return variables


_T_co = TypeVar("_T_co", covariant=True)


def batched(iterable: Iterable[_T_co], n: int) -> Iterator[tuple[_T_co, ...]]:
    """Batch data into tuples of length n. The last batch may be shorter."""
    # Based on: https://docs.python.org/3/library/itertools.html#itertools-recipes
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("Batch size must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def batched_it(iterable: Iterable[_T_co], n: int) -> Iterator[Iterator[_T_co]]:
    """Batch data into iterators of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("Batch size must be at least one")
    it = iter(iterable)
    while True:
        chunk_it = islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield chain((first_el,), chunk_it)


def flatten(items):
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from item
        else:
            yield item


def retry_with_backoff(retries=5, backoff_sec=1, errors=(Exception,)):
    def retry(f):
        def wrapper(*args, **kwargs):
            num_tried = 0
            while True:
                try:
                    return f(*args, **kwargs)
                except errors:
                    if num_tried == retries:
                        raise
                    sleep = (
                        backoff_sec * 2** num_tried + random.uniform(0, 1)  # noqa: S311
                    )
                    logger.exception(
                        "Error in %s, retrying in %ds, attempt %d",
                        f.__name__,
                        sleep,
                        num_tried,
                    )
                    time.sleep(sleep)
                    num_tried += 1

        return wrapper

    return retry


def determine_processes(parallel: Optional[Union[bool, int]]) -> Union[bool, int]:
    if parallel is None and os.environ.get("DATACHAIN_SETTINGS_PARALLEL") is not None:
        parallel = int(os.environ["DATACHAIN_SETTINGS_PARALLEL"])
    if parallel is None or parallel is False:
        return False
    if parallel is True:
        return True
    if parallel == 0:
        return False
    if parallel < 0:
        return True
    return parallel


def get_env_list(
    key: str, default: Optional[Sequence] = None, sep: str = ","
) -> Optional[Sequence[str]]:
    try:
        str_val = os.environ[key]
    except KeyError:
        return default
    return str_val.split(sep=sep)


def show_df(
    df: "pd.DataFrame", collapse_columns: bool = True, system_columns: bool = False
) -> None:
    import pandas as pd

    if df.empty:
        return

    options: list[Any] = ["display.show_dimensions", False, "display.min_rows", 0]
    if not collapse_columns:
        options.extend(("display.max_columns", None))  # show all columns
        options.extend(("display.max_colwidth", None))  # do not truncate cells
        options.extend(("display.width", None))  #  do not truncate table

    if not system_columns:
        df.drop(
            columns=[
                "dir_type",
                "etag",
                "is_latest",
                "last_modified",
                "size",
                "version",
            ],
            inplace=True,
            errors="ignore",
        )

    with pd.option_context("display.max_rows", None, *options):  # show all rows
        print(df)


def show_records(
    records: Optional[list[dict]],
    collapse_columns: bool = False,
    system_columns: bool = False,
    hidden_fields: Optional[list[str]] = None,
) -> None:
    import pandas as pd

    if not records:
        return

    df = pd.DataFrame.from_records(records)
    if hidden_fields:
        df = df.drop(columns=hidden_fields, errors="ignore")
    return show_df(df, collapse_columns=collapse_columns, system_columns=system_columns)


class JSONSerialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return list(obj[:1024])
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)

        return super().default(obj)


def inside_colab() -> bool:
    try:
        from google import colab  # noqa: F401
    except ImportError:
        return False
    return True


def inside_notebook() -> bool:
    if inside_colab():
        return True

    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except NameError:
        return False

    if shell == "ZMQInteractiveShell":
        try:
            import IPython

            return IPython.__version__ >= "6.0.0"
        except ImportError:
            return False

    return False


def get_all_subclasses(cls):
    """Return all subclasses of a given class.
    Can return duplicates due to multiple inheritance."""
    for subclass in cls.__subclasses__():
        yield from get_all_subclasses(subclass)
        yield subclass


def filtered_cloudpickle_dumps(obj: Any) -> bytes:
    """Equivalent to cloudpickle.dumps, but this supports Pydantic models."""
    model_namespaces = {}

    with io.BytesIO() as f:
        pickler = cloudpickle.CloudPickler(f)

        for model_class in get_all_subclasses(BaseModel):
            # This "is not None" check is needed, because due to multiple inheritance,
            # it is theoretically possible to get the same class twice from
            # get_all_subclasses.
            if model_class.__pydantic_parent_namespace__ is not None:
                # __pydantic_parent_namespace__ can contain many unnecessary and
                # unpickleable entities, so should be removed for serialization.
                model_namespaces[model_class] = (
                    model_class.__pydantic_parent_namespace__
                )
                model_class.__pydantic_parent_namespace__ = None

        try:
            pickler.dump(obj)
            return f.getvalue()
        finally:
            for model_class, namespace in model_namespaces.items():
                # Restore original __pydantic_parent_namespace__ locally.
                model_class.__pydantic_parent_namespace__ = namespace


def get_datachain_executable() -> list[str]:
    if datachain_exec_path := os.getenv("DATACHAIN_EXEC_PATH"):
        return [datachain_exec_path]
    return [sys.executable, "-m", "datachain"]


def uses_glob(path: str) -> bool:
    """Checks if some URI path has glob syntax in it"""
    return glob.has_magic(os.path.basename(os.path.normpath(path)))


def env2bool(var, undefined=False):
    """
    undefined: return value if env var is unset
    """
    var = os.getenv(var, None)
    if var is None:
        return undefined
    return bool(re.search("1|y|yes|true", var, flags=re.IGNORECASE))


def nested_dict_path_set(
    data: dict[str, Any], path: Sequence[str], value: Any
) -> dict[str, Any]:
    """Sets a value inside a nested dict based on the list of dict keys as a path,
    and will create sub-dicts as needed to set the value."""
    sub_data = data
    for element in path[:-1]:
        if element not in sub_data:
            sub_data[element] = {}
        sub_data = sub_data[element]
    sub_data[path[len(path) - 1]] = value
    return data


def row_to_nested_dict(
    headers: Iterable[Sequence[str]], row: Iterable[Any]
) -> dict[str, Any]:
    """Converts a row to a nested dict based on the provided headers."""
    result: dict[str, Any] = {}
    for h, v in zip(headers, row):
        nested_dict_path_set(result, h, v)
    return result


@contextmanager
def safe_closing(thing: T) -> Iterator[T]:
    try:
        yield thing
    finally:
        if hasattr(thing, "close"):
            thing.close()
