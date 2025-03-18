from datachain.client import Client


def test_anon_url():
    client = Client.get_client("gs://foo", None, anon=True)
    assert client.url("bar") == "https://storage.googleapis.com/foo/bar"


def test_anon_versioned_url():
    client = Client.get_client("gs://foo", None, anon=True)
    assert (
        client.url("bar", version_id="1234566")
        == "https://storage.googleapis.com/foo/bar?generation=1234566"
    )
