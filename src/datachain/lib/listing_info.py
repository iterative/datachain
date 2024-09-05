from datetime import datetime, timedelta, timezone
from typing import Optional

from datachain.client import Client
from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.listing import LISTING_PREFIX, LISTING_TTL


class ListingInfo(DatasetInfo):
    @property
    def uri(self) -> str:
        return self.name.removeprefix(LISTING_PREFIX)

    @property
    def storage_uri(self) -> str:
        client, _ = Client.parse_url(self.uri, None)  # type: ignore[arg-type]
        return client.uri

    @property
    def expires(self) -> Optional[datetime]:
        if not self.finished_at:
            return None
        return self.finished_at + timedelta(seconds=LISTING_TTL)

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires if self.expires else False

    @property
    def last_inserted_at(self):
        # TODO we need to add updated_at to dataset version or explicit last_inserted_at
        raise NotImplementedError
