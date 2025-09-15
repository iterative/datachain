from unittest.mock import patch

from pydantic import BaseModel

import datachain as dc


class CustomFeature(BaseModel):
    sqrt: float
    my_name: str


def test_read_values():
    # hash from read values is currently inconsietent (evety time we get different
    # value).
    assert dc.read_values(num=[1, 2, 3]).hash()


def test_read_storage():
    with patch("datachain.lib.dc.storage.get_listing") as mock_listing:
        mock_listing.return_value = ("lst__s3://my-bucket", "", "", True)

        assert dc.read_storage("s3://my-bucket").hash() == (
            "c38b6f4ebd7f0160d9f900016aad1e6781acd463f042588cfe793e9d189a8a0e"
        )


def test_read_dataset(test_session):
    dc.read_values(num=[1, 2, 3], session=test_session).save("cats")
    assert dc.read_dataset(
        name="cats", version="1.0.0", session=test_session
    ).hash() == ("54634c934f1d0d03bdd9409d0dcff3a6261921a78a0ebce4752bf96a16b99604")
