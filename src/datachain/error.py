class DataChainError(RuntimeError):
    pass


class NotFoundError(Exception):
    pass


class DatasetNotFoundError(NotFoundError):
    pass


class DatasetInvalidVersionError(Exception):
    pass


class StorageNotFoundError(NotFoundError):
    pass


class PendingIndexingError(Exception):
    """An indexing operation is already in progress."""


class QueryScriptCompileError(Exception):
    pass


class QueryScriptRunError(Exception):
    """Error raised by `subprocess.run`.

    Attributes:
        message      Explanation of the error
        return_code  Code returned by the subprocess
        output       STDOUT + STDERR output of the subprocess
    """

    def __init__(self, message: str, return_code: int = 0, output: str = ""):
        self.message = message
        self.return_code = return_code
        self.output = output
        super().__init__(self.message)


class QueryScriptDatasetNotFound(QueryScriptRunError):  # noqa: N818
    pass


class QueryScriptCancelError(QueryScriptRunError):
    pass


class ClientError(RuntimeError):
    def __init__(self, message, error_code=None):
        super().__init__(message)
        # error code from the cloud itself
        self.error_code = error_code


class TableMissingError(DataChainError):
    pass
