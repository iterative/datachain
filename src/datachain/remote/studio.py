import json
import logging
import os
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone
from struct import unpack
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
)

from datachain.config import Config
from datachain.dataset import DatasetStats
from datachain.error import DataChainError
from datachain.utils import STUDIO_URL, retry_with_backoff

T = TypeVar("T")
LsData = Optional[list[dict[str, Any]]]
DatasetInfoData = Optional[dict[str, Any]]
DatasetStatsData = Optional[DatasetStats]
DatasetRowsData = Optional[Iterable[dict[str, Any]]]
DatasetExportStatus = Optional[dict[str, Any]]
DatasetExportSignedUrls = Optional[list[str]]


logger = logging.getLogger("datachain")

DATASET_ROWS_CHUNK_SIZE = 8192


def _is_server_error(status_code: int) -> bool:
    return str(status_code).startswith("5")


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
                "Studio token is not set. Use `datachain studio login` "
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
                "Use `datachain studio team <team_name>` "
                "or environment variable `DVC_STUDIO_TEAM` to set it."
                "You can also set it in the config file as team under studio."
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

    def _send_request_msgpack(self, route: str, data: dict[str, Any]) -> Response[Any]:
        import msgpack
        import requests

        response = requests.post(
            f"{self.url}/{route}",
            json={**data, "team_name": self.team},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"token {self.token}",
            },
            timeout=self.timeout,
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

    @retry_with_backoff(retries=5)
    def _send_request(self, route: str, data: dict[str, Any]) -> Response[Any]:
        """
        Function that communicate Studio API.
        It will raise an exception, and try to retry, if 5xx status code is
        returned, or if ConnectionError or Timeout exceptions are thrown from
        requests lib
        """
        import requests

        response = requests.post(
            f"{self.url}/{route}",
            json={**data, "team_name": self.team},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"token {self.token}",
            },
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
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

    def ls(self, paths: Iterable[str]) -> Iterator[tuple[str, Response[LsData]]]:
        # TODO: change LsData (response.data value) to be list of lists
        # to handle cases where a path will be expanded (i.e. globs)
        response: Response[LsData]
        for path in paths:
            response = self._send_request_msgpack("datachain/ls", {"source": path})
            yield path, response

    def ls_datasets(self) -> Response[LsData]:
        return self._send_request("datachain/ls-datasets", {})

    def edit_dataset(
        self,
        name: str,
        new_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[list[str]] = None,
    ) -> Response[DatasetInfoData]:
        body = {
            "dataset_name": name,
        }

        if new_name is not None:
            body["new_name"] = new_name

        if description is not None:
            body["description"] = description

        if labels is not None:
            body["labels"] = labels  # type: ignore[assignment]

        return self._send_request(
            "datachain/edit-dataset",
            body,
        )

    def rm_dataset(
        self,
        name: str,
        version: Optional[int] = None,
        force: Optional[bool] = False,
    ) -> Response[DatasetInfoData]:
        return self._send_request(
            "datachain/rm-dataset",
            {
                "dataset_name": name,
                "version": version,
                "force": force,
            },
        )

    def dataset_info(self, name: str) -> Response[DatasetInfoData]:
        def _parse_dataset_info(dataset_info):
            _parse_dates(dataset_info, ["created_at", "finished_at"])
            for version in dataset_info.get("versions"):
                _parse_dates(version, ["created_at"])

            return dataset_info

        response = self._send_request("datachain/dataset-info", {"dataset_name": name})
        if response.ok:
            response.data = _parse_dataset_info(response.data)
        return response

    def dataset_rows_chunk(
        self, name: str, version: int, offset: int
    ) -> Response[DatasetRowsData]:
        req_data = {"dataset_name": name, "dataset_version": version}
        return self._send_request_msgpack(
            "datachain/dataset-rows",
            {**req_data, "offset": offset, "limit": DATASET_ROWS_CHUNK_SIZE},
        )

    def dataset_stats(self, name: str, version: int) -> Response[DatasetStatsData]:
        response = self._send_request(
            "datachain/dataset-stats",
            {"dataset_name": name, "dataset_version": version},
        )
        if response.ok:
            response.data = DatasetStats(**response.data)
        return response

    def export_dataset_table(
        self, name: str, version: int
    ) -> Response[DatasetExportSignedUrls]:
        return self._send_request(
            "datachain/dataset-export",
            {"dataset_name": name, "dataset_version": version},
        )

    def dataset_export_status(
        self, name: str, version: int
    ) -> Response[DatasetExportStatus]:
        return self._send_request(
            "datachain/dataset-export-status",
            {"dataset_name": name, "dataset_version": version},
        )
