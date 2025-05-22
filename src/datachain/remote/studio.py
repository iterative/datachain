import base64
import json
import logging
import os
from collections.abc import AsyncIterator, Iterable, Iterator
from datetime import datetime, timedelta, timezone
from struct import unpack
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
)
from urllib.parse import urlparse, urlunparse

import websockets
from requests.exceptions import HTTPError, Timeout

from datachain.config import Config
from datachain.error import DataChainError
from datachain.utils import STUDIO_URL, retry_with_backoff

T = TypeVar("T")
LsData = Optional[list[dict[str, Any]]]
DatasetInfoData = Optional[dict[str, Any]]
DatasetRowsData = Optional[Iterable[dict[str, Any]]]
DatasetJobVersionsData = Optional[dict[str, Any]]
DatasetExportStatus = Optional[dict[str, Any]]
DatasetExportSignedUrls = Optional[list[str]]
FileUploadData = Optional[dict[str, Any]]
JobData = Optional[dict[str, Any]]
JobListData = dict[str, Any]
logger = logging.getLogger("datachain")

DATASET_ROWS_CHUNK_SIZE = 8192


def _is_server_error(status_code: int) -> bool:
    return str(status_code).startswith("5")


def is_token_set() -> bool:
    return (
        bool(os.environ.get("DVC_STUDIO_TOKEN"))
        or Config().read().get("studio", {}).get("token") is not None
    )


def _parse_dates(obj: dict, date_fields: list[str]):
    """
    Function that converts string ISO dates to datetime.datetime instances in object
    """
    for date_field in date_fields:
        if obj.get(date_field):
            obj[date_field] = datetime.fromisoformat(obj[date_field])


class Response(Generic[T]):
    def __init__(self, data: T, ok: bool, message: str) -> None:
        self.data = data
        self.ok = ok
        self.message = message

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(ok={self.ok}, data={self.data}"
            f", message={self.message})"
        )


class StudioClient:
    def __init__(self, timeout: float = 3600.0, team: Optional[str] = None) -> None:
        self._check_dependencies()
        self.timeout = timeout
        self._config = None
        self._team = team

    @property
    def token(self) -> str:
        token = os.environ.get("DVC_STUDIO_TOKEN") or self.config.get("token")

        if not token:
            raise DataChainError(
                "Studio token is not set. Use `datachain auth login` "
                "or environment variable `DVC_STUDIO_TOKEN` to set it."
            )

        return token

    @property
    def url(self) -> str:
        return (
            os.environ.get("DVC_STUDIO_URL") or self.config.get("url") or STUDIO_URL
        ) + "/api"

    @property
    def config(self) -> dict:
        if self._config is None:
            self._config = Config().read().get("studio", {})
        return self._config  # type: ignore [return-value]

    @property
    def team(self) -> str:
        if self._team is None:
            self._team = self._get_team()
        return self._team

    def _get_team(self) -> str:
        team = os.environ.get("DVC_STUDIO_TEAM") or self.config.get("team")

        if not team:
            raise DataChainError(
                "Studio team is not set. "
                "Use `datachain auth team <team_name>` "
                "or environment variable `DVC_STUDIO_TEAM` to set it. "
                "You can also set `studio.team` in the config file."
            )

        return team

    def _check_dependencies(self) -> None:
        try:
            import msgpack  # noqa: F401
            import requests  # noqa: F401
        except ImportError as exc:
            raise Exception(
                f"Missing dependency: {exc.name}\n"
                "To install run:\n"
                "\tpip install 'datachain[remote]'"
            ) from None

    def _send_request_msgpack(
        self, route: str, data: dict[str, Any], method: Optional[str] = "POST"
    ) -> Response[Any]:
        import msgpack
        import requests

        kwargs = (
            {"params": {**data, "team_name": self.team}}
            if method == "GET"
            else {"json": {**data, "team_name": self.team}}
        )

        response = requests.request(
            method=method,  # type: ignore[arg-type]
            url=f"{self.url}/{route}",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"token {self.token}",
            },
            timeout=self.timeout,
            **kwargs,  # type: ignore[arg-type]
        )
        ok = response.ok
        if not ok:
            if response.status_code == 403:
                message = f"Not authorized for the team {self.team}"
                raise DataChainError(message)
            logger.error("Got bad response from Studio")

        content = msgpack.unpackb(response.content, ext_hook=self._unpacker_hook)
        response_data = content.get("data")
        if ok and response_data is None:
            message = "Indexing in progress"
        else:
            message = content.get("message", "")
        return Response(response_data, ok, message)

    @retry_with_backoff(retries=3, errors=(HTTPError, Timeout))
    def _send_request(
        self, route: str, data: dict[str, Any], method: Optional[str] = "POST"
    ) -> Response[Any]:
        """
        Function that communicate Studio API.
        It will raise an exception, and try to retry, if 5xx status code is
        returned, or if Timeout exceptions is thrown from the requests lib
        """
        import requests

        kwargs = (
            {"params": {**data, "team_name": self.team}}
            if method == "GET"
            else {"json": {**data, "team_name": self.team}}
        )

        response = requests.request(
            method=method,  # type: ignore[arg-type]
            url=f"{self.url}/{route}",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"token {self.token}",
            },
            timeout=self.timeout,
            **kwargs,  # type: ignore[arg-type]
        )
        try:
            response.raise_for_status()
        except HTTPError:
            if _is_server_error(response.status_code):
                # going to retry
                raise

        ok = response.ok
        try:
            data = json.loads(response.content.decode("utf-8"))
        except json.decoder.JSONDecodeError:
            data = {}

        if not ok:
            if response.status_code == 403:
                message = f"Not authorized for the team {self.team}"
            else:
                message = data.get("message", "")
        else:
            message = ""

        return Response(data, ok, message)

    @staticmethod
    def _unpacker_hook(code, data):
        import msgpack

        if code == 42:  # for parsing datetime objects
            has_timezone = False
            timezone_offset = None
            if len(data) == 8:
                # we send only timestamp without timezone if data is 8 bytes
                values = unpack("!d", data)
            else:
                has_timezone = True
                values = unpack("!dl", data)

            timestamp = values[0]
            if has_timezone:
                timezone_offset = values[1]
                return datetime.fromtimestamp(
                    timestamp, timezone(timedelta(seconds=timezone_offset))
                )
            return datetime.fromtimestamp(timestamp)  # noqa: DTZ006

        return msgpack.ExtType(code, data)

    async def tail_job_logs(self, job_id: str) -> AsyncIterator[dict]:
        """
        Follow job logs via websocket connection.

        Args:
            job_id: ID of the job to follow logs for

        Yields:
            Dict containing either job status updates or log messages
        """
        parsed_url = urlparse(self.url)
        ws_url = urlunparse(
            parsed_url._replace(scheme="wss" if parsed_url.scheme == "https" else "ws")
        )
        ws_url = f"{ws_url}/logs/follow/?job_id={job_id}&team_name={self.team}"

        async with websockets.connect(
            ws_url,
            additional_headers={"Authorization": f"token {self.token}"},
        ) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)

                    # Yield the parsed message data
                    yield data

                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:  # noqa: BLE001
                    logger.error("Error receiving websocket message: %s", e)
                    break

    def ls(self, paths: Iterable[str]) -> Iterator[tuple[str, Response[LsData]]]:
        # TODO: change LsData (response.data value) to be list of lists
        # to handle cases where a path will be expanded (i.e. globs)
        response: Response[LsData]
        for path in paths:
            response = self._send_request_msgpack("datachain/ls", {"source": path})
            yield path, response

    def ls_datasets(self, prefix: Optional[str] = None) -> Response[LsData]:
        return self._send_request(
            "datachain/datasets", {"prefix": prefix}, method="GET"
        )

    def edit_dataset(
        self,
        name: str,
        new_name: Optional[str] = None,
        description: Optional[str] = None,
        attrs: Optional[list[str]] = None,
    ) -> Response[DatasetInfoData]:
        body = {
            "new_name": new_name,
            "dataset_name": name,
            "description": description,
            "attrs": attrs,
        }

        return self._send_request(
            "datachain/datasets",
            body,
        )

    def rm_dataset(
        self,
        name: str,
        version: Optional[str] = None,
        force: Optional[bool] = False,
    ) -> Response[DatasetInfoData]:
        return self._send_request(
            "datachain/datasets",
            {
                "dataset_name": name,
                "dataset_version": version,
                "force": force,
            },
            method="DELETE",
        )

    def dataset_info(self, name: str) -> Response[DatasetInfoData]:
        def _parse_dataset_info(dataset_info):
            _parse_dates(dataset_info, ["created_at", "finished_at"])
            for version in dataset_info.get("versions"):
                _parse_dates(version, ["created_at"])

            return dataset_info

        response = self._send_request(
            "datachain/datasets/info", {"dataset_name": name}, method="GET"
        )
        if response.ok:
            response.data = _parse_dataset_info(response.data)
        return response

    def dataset_rows_chunk(
        self, name: str, version: str, offset: int
    ) -> Response[DatasetRowsData]:
        req_data = {"dataset_name": name, "dataset_version": version}
        return self._send_request_msgpack(
            "datachain/datasets/rows",
            {**req_data, "offset": offset, "limit": DATASET_ROWS_CHUNK_SIZE},
            method="GET",
        )

    def dataset_job_versions(self, job_id: str) -> Response[DatasetJobVersionsData]:
        return self._send_request(
            "datachain/datasets/dataset_job_versions",
            {"job_id": job_id},
            method="GET",
        )

    def export_dataset_table(
        self, name: str, version: str
    ) -> Response[DatasetExportSignedUrls]:
        return self._send_request(
            "datachain/datasets/export",
            {"dataset_name": name, "dataset_version": version},
            method="GET",
        )

    def dataset_export_status(
        self, name: str, version: str
    ) -> Response[DatasetExportStatus]:
        return self._send_request(
            "datachain/datasets/export-status",
            {"dataset_name": name, "dataset_version": version},
            method="GET",
        )

    def upload_file(self, content: bytes, file_name: str) -> Response[FileUploadData]:
        data = {
            "file_content": base64.b64encode(content).decode("utf-8"),
            "file_name": file_name,
        }
        return self._send_request("datachain/upload-file", data)

    def create_job(
        self,
        query: str,
        query_type: str,
        environment: Optional[str] = None,
        workers: Optional[int] = None,
        query_name: Optional[str] = None,
        files: Optional[list[str]] = None,
        python_version: Optional[str] = None,
        requirements: Optional[str] = None,
        repository: Optional[str] = None,
        priority: Optional[int] = None,
    ) -> Response[JobData]:
        data = {
            "query": query,
            "query_type": query_type,
            "environment": environment,
            "workers": workers,
            "query_name": query_name,
            "files": files,
            "python_version": python_version,
            "requirements": requirements,
            "repository": repository,
            "priority": priority,
        }
        return self._send_request("datachain/job", data)

    def get_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> Response[JobListData]:
        return self._send_request(
            "datachain/jobs",
            {"status": status, "limit": limit} if status else {"limit": limit},
            method="GET",
        )

    def cancel_job(
        self,
        job_id: str,
    ) -> Response[JobData]:
        url = f"datachain/job/{job_id}/cancel"
        return self._send_request(url, data={}, method="POST")
