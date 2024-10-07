import atexit
import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from datachain.catalog import get_catalog
from datachain.error import TableMissingError

if TYPE_CHECKING:
    from datachain.catalog import Catalog

logger = logging.getLogger("datachain")


class Session:
    """
    Session is a context that keeps track of temporary DataChain datasets for a proper
    cleanup. By default, a global session is created.

    Temporary or ephemeral datasets are the ones created without specified name.
    They are useful for optimization purposes and should be automatically removed.

    Temp dataset has specific name format:
        "session_<name>_<session_uuid>_<dataset_uuid>"
    The <name> suffix is optional. Both <uuid>s are auto-generated.

    Temp dataset examples:
        session_myname_624b41_48e8b4
        session_4b962d_2a5dff

    Parameters:

    name (str): The name of the session. Only latters and numbers are supported.
           It can be empty.
    catalog (Catalog): Catalog object.
    """

    GLOBAL_SESSION_CTX: Optional["Session"] = None
    GLOBAL_SESSION: Optional["Session"] = None
    ORIGINAL_EXCEPT_HOOK = None

    DATASET_PREFIX = "session_"
    GLOBAL_SESSION_NAME = "global"
    SESSION_UUID_LEN = 6
    TEMP_TABLE_UUID_LEN = 6

    def __init__(
        self,
        name="",
        catalog: Optional["Catalog"] = None,
        client_config: Optional[dict] = None,
        in_memory: bool = False,
    ):
        if re.match(r"^[0-9a-zA-Z]+$", name) is None:
            raise ValueError(
                f"Session name can contain only letters or numbers - '{name}' given."
            )

        if not name:
            name = self.GLOBAL_SESSION_NAME

        session_uuid = uuid4().hex[: self.SESSION_UUID_LEN]
        self.name = f"{name}_{session_uuid}"
        self.job_id = os.getenv("DATACHAIN_JOB_ID") or str(uuid4())
        self.is_new_catalog = not catalog
        self.catalog = catalog or get_catalog(
            client_config=client_config, in_memory=in_memory
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._cleanup_created_versions(self.name)

        self._cleanup_temp_datasets()
        if self.is_new_catalog:
            self.catalog.metastore.close_on_exit()
            self.catalog.warehouse.close_on_exit()
            self.catalog.id_generator.close_on_exit()

    def generate_temp_dataset_name(self) -> str:
        return self.get_temp_prefix() + uuid4().hex[: self.TEMP_TABLE_UUID_LEN]

    def get_temp_prefix(self) -> str:
        return f"{self.DATASET_PREFIX}{self.name}_"

    def _cleanup_temp_datasets(self) -> None:
        prefix = self.get_temp_prefix()
        try:
            for dataset in list(self.catalog.metastore.list_datasets_by_prefix(prefix)):
                self.catalog.remove_dataset(dataset.name, force=True)
        # suppress error when metastore has been reset during testing
        except TableMissingError:
            pass

    def _cleanup_created_versions(self, job_id: str) -> None:
        versions = self.catalog.metastore.get_job_dataset_versions(job_id)
        if not versions:
            return

        datasets = {}
        for dataset_name, version in versions:
            if dataset_name not in datasets:
                datasets[dataset_name] = self.catalog.get_dataset(dataset_name)
            dataset = datasets[dataset_name]
            logger.info(
                "Removing dataset version %s@%s due to exception", dataset_name, version
            )
            self.catalog.remove_dataset_version(dataset, version)

    @classmethod
    def get(
        cls,
        session: Optional["Session"] = None,
        catalog: Optional["Catalog"] = None,
        client_config: Optional[dict] = None,
        in_memory: bool = False,
    ) -> "Session":
        """Creates a Session() object from a catalog.

        Parameters:
            session (Session): Optional Session(). If not provided a new session will
                    be created. It's needed mostly for simplie API purposes.
            catalog (Catalog): Optional catalog. By default a new catalog is created.
        """
        if session:
            return session

        if cls.GLOBAL_SESSION is None:
            cls.GLOBAL_SESSION_CTX = Session(
                cls.GLOBAL_SESSION_NAME,
                catalog,
                client_config=client_config,
                in_memory=in_memory,
            )
            cls.GLOBAL_SESSION = cls.GLOBAL_SESSION_CTX.__enter__()

            atexit.register(cls._global_cleanup)
            cls.ORIGINAL_EXCEPT_HOOK = sys.excepthook
            sys.excepthook = cls.except_hook

        return cls.GLOBAL_SESSION

    @staticmethod
    def except_hook(exc_type, exc_value, exc_traceback):
        Session._global_cleanup()
        if Session.GLOBAL_SESSION_CTX is not None:
            job_id = Session.GLOBAL_SESSION_CTX.job_id
            Session.GLOBAL_SESSION_CTX._cleanup_created_versions(job_id)

        if Session.ORIGINAL_EXCEPT_HOOK:
            Session.ORIGINAL_EXCEPT_HOOK(exc_type, exc_value, exc_traceback)

    @classmethod
    def cleanup_for_tests(cls):
        if cls.GLOBAL_SESSION_CTX is not None:
            cls.GLOBAL_SESSION_CTX.__exit__(None, None, None)
            cls.GLOBAL_SESSION = None
            cls.GLOBAL_SESSION_CTX = None
            atexit.unregister(cls._global_cleanup)

        if cls.ORIGINAL_EXCEPT_HOOK:
            sys.excepthook = cls.ORIGINAL_EXCEPT_HOOK

    @staticmethod
    def _global_cleanup():
        if Session.GLOBAL_SESSION_CTX is not None:
            Session.GLOBAL_SESSION_CTX.__exit__(None, None, None)
