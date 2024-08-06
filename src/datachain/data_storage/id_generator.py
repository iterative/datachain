import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Column, Integer, Table, Text

from datachain.data_storage.serializer import Serializable

if TYPE_CHECKING:
    from sqlalchemy.schema import SchemaItem

    from datachain.data_storage.db_engine import DatabaseEngine


logger = logging.getLogger("datachain")


class AbstractIDGenerator(ABC, Serializable):
    """
    Abstract ID Generator class. This class is responsible for generating
    unique IDs for each prefix (e.g. S3 bucket or dataset).
    """

    @abstractmethod
    def clone(self) -> "AbstractIDGenerator":
        """Clones AbstractIDGenerator implementation."""

    def init(self) -> None:
        """Initialize ID generator."""

    def cleanup_for_tests(self):
        """Cleanup for tests."""

    def close(self) -> None:
        """Closes any active database connections."""

    def close_on_exit(self) -> None:
        """Closes any active database or HTTP connections, called on Session exit or
        for test cleanup only, as some ID Generator implementations may handle this
        differently.
        """
        self.close()

    @abstractmethod
    def init_id(self, uri: str) -> None:
        """Initializes the ID generator for the given URI with zero last_id."""

    @abstractmethod
    def get_next_ids(self, uri: str, count: int) -> range:
        """Returns a range of IDs for the given URI."""

    def get_next_id(self, uri: str) -> int:
        """Returns the next ID for the given URI."""
        return self.get_next_ids(uri, 1)[0]

    def delete_uri(self, uri: str):
        """Deletes the given URI."""
        self.delete_uris([uri])

    def delete_uris(self, uris: Iterable[str]):
        """Deletes the given URIs."""


class AbstractDBIDGenerator(AbstractIDGenerator):
    """
    Abstract ID Generator class, to be implemented by any Database Adapters
    for a specific database system. This class is responsible for generating
    unique IDs for each prefix (e.g. S3 bucket or dataset) and storing them
    in a database. It is also responsible for initializing the database
    and creating the necessary tables.
    """

    _db: "DatabaseEngine"
    _table_prefix: Optional[str] = None
    _skip_db_init: bool = False
    _base_table_name = "id_generator"

    def __init__(
        self,
        db: "DatabaseEngine",
        table_prefix: Optional[str] = None,
        skip_db_init: bool = False,
    ):
        self._db = db
        self._table_prefix = table_prefix
        self._skip_db_init = skip_db_init
        if db and not skip_db_init:
            self.init()

    @abstractmethod
    def clone(self) -> "AbstractDBIDGenerator":
        """Clones AbstractIDGenerator implementation."""

    def close(self) -> None:
        """Closes any active database connections."""
        self.db.close()

    @property
    def db(self) -> "DatabaseEngine":
        return self._db

    @property
    def _columns(self) -> list["SchemaItem"]:
        return [
            Column("uri", Text, primary_key=True, nullable=False),
            # This is the last id used (and starts at zero if no ids have been used)
            Column("last_id", Integer, nullable=False),
        ]

    @cached_property
    def _table(self) -> Table:
        table_name = self._base_table_name
        if self._table_prefix:
            table_name = f"{self._table_prefix}_{table_name}"
        return Table(table_name, self.db.metadata, *self._columns, extend_existing=True)

    def init(self) -> None:
        self.db.create_table(self._table, if_not_exists=True)

    def cleanup_for_tests(self):
        """Cleanup for tests."""
        self.db.drop_table(self._table, if_exists=True)

    @abstractmethod
    def init_id(self, uri: str) -> None:
        """Initializes the ID generator for the given URI with zero last_id."""

    @abstractmethod
    def get_next_ids(self, uri: str, count: int) -> range:
        """Returns a range of IDs for the given URI."""

    def delete_uris(self, uris: Iterable[str]):
        """Deletes the given URIs from the database."""
        self.db.execute(self._table.delete().where(self._table.c.uri.in_(uris)))
