from datachain.client import Client


def test_anon_url(mocker):
    def sign(*args, **kwargs):
        raise AttributeError(
            "you need a private key to sign credentials."
            "the credentials you are currently using"
            " <class 'google.oauth2.credentials.Credentials'> just contains a token."
            " see https://googleapis.dev/python/google-api-core/latest/auth.html"
            "#setting-up-a-service-account for more details."
        )

    mocker.patch("gcsfs.GCSFileSystem.sign", side_effect=sign)

    client = Client.get_client("gs://foo", None, anon=True)
    assert client.url("bar") == "https://storage.googleapis.com/foo/bar"
