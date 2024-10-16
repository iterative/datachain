import pytest
from dvc_studio_client.auth import AuthorizationExpiredError

from datachain.cli import main
from datachain.config import Config
from datachain.studio import POST_LOGIN_MESSAGE
from datachain.utils import ENV_DATACHAIN_GLOBAL_CONFIG_DIR, STUDIO_URL


@pytest.fixture(autouse=True)
def global_config_dir(monkeypatch, tmp_path_factory):
    monkeypatch.setenv(
        ENV_DATACHAIN_GLOBAL_CONFIG_DIR, str(tmp_path_factory.mktemp("studio-login"))
    )


def test_studio_login_token_check_failed(mocker):
    mocker.patch(
        "dvc_studio_client.auth.get_access_token",
        side_effect=AuthorizationExpiredError,
    )
    assert main(["studio", "login"]) == 1


def test_studio_login_success(mocker):
    mocker.patch(
        "dvc_studio_client.auth.get_access_token",
        return_value=("token_name", "isat_access_token"),
    )

    assert main(["studio", "login"]) == 0

    config = Config().read()
    assert config["studio"]["token"] == "isat_access_token"  # noqa: S105 # nosec B105
    assert config["studio"]["url"] == STUDIO_URL


def test_studio_login_arguments(mocker):
    mock = mocker.patch(
        "dvc_studio_client.auth.get_access_token",
        return_value=("token_name", "isat_access_token"),
    )

    assert (
        main(
            [
                "studio",
                "login",
                "--name",
                "token_name",
                "--hostname",
                "https://example.com",
                "--scopes",
                "experiments",
                "--no-open",
            ]
        )
        == 0
    )

    mock.assert_called_with(
        token_name="token_name",  #  noqa: S106
        hostname="https://example.com",
        scopes="experiments",
        client_name="Datachain",
        open_browser=False,
        post_login_message=POST_LOGIN_MESSAGE,
    )


def test_studio_logout():
    with Config("global").edit() as conf:
        conf["studio"] = {"token": "isat_access_token"}

    assert main(["studio", "logout"]) == 0
    config = Config("global").read()
    assert "token" not in config["studio"]

    assert main(["studio", "logout"]) == 1


def test_studio_token(capsys):
    with Config("global").edit() as conf:
        conf["studio"] = {"token": "isat_access_token"}

    assert main(["studio", "token"]) == 0
    assert capsys.readouterr().out == "isat_access_token\n"

    with Config("global").edit() as conf:
        del conf["studio"]["token"]

    assert main(["studio", "token"]) == 1
