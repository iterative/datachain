from dvc_studio_client.auth import AuthorizationExpiredError
from tabulate import tabulate

from datachain.cli import main
from datachain.config import Config, ConfigLevel
from datachain.studio import POST_LOGIN_MESSAGE
from datachain.utils import STUDIO_URL


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
        client_name="DataChain",
        open_browser=False,
        post_login_message=POST_LOGIN_MESSAGE,
    )


def test_studio_logout():
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"token": "isat_access_token"}

    assert main(["studio", "logout"]) == 0
    config = Config(ConfigLevel.GLOBAL).read()
    assert "token" not in config["studio"]

    assert main(["studio", "logout"]) == 1


def test_studio_token(capsys):
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"token": "isat_access_token"}

    assert main(["studio", "token"]) == 0
    assert capsys.readouterr().out == "isat_access_token\n"

    with Config(ConfigLevel.GLOBAL).edit() as conf:
        del conf["studio"]["token"]

    assert main(["studio", "token"]) == 1


def test_studio_ls_datasets(capsys, studio_datasets):
    assert main(["studio", "datasets"]) == 0
    out = capsys.readouterr().out

    expected_rows = [
        {"Name": "dogs", "Version": "1"},
        {"Name": "dogs", "Version": "2"},
        {
            "Name": "cats",
            "Version": "1",
        },
        {"Name": "both", "Version": "1"},
    ]
    assert out.strip() == tabulate(expected_rows, headers="keys")


def test_studio_team_local():
    assert main(["studio", "team", "team_name"]) == 0
    config = Config(ConfigLevel.LOCAL).read()
    assert config["studio"]["team"] == "team_name"


def test_studio_team_global():
    assert main(["studio", "team", "team_name", "--global"]) == 0
    config = Config(ConfigLevel.GLOBAL).read()
    assert config["studio"]["team"] == "team_name"


def test_studio_datasets(capsys, studio_datasets, mocker):
    def list_datasets_local(_):
        yield "local", 1
        yield "both", 1

    mocker.patch("datachain.cli.list_datasets_local", side_effect=list_datasets_local)
    local_rows = [
        {"Name": "both", "Version": "1"},
        {"Name": "local", "Version": "1"},
    ]
    local_output = tabulate(local_rows, headers="keys")

    studio_rows = [
        {"Name": "both", "Version": "1"},
        {
            "Name": "cats",
            "Version": "1",
        },
        {"Name": "dogs", "Version": "1"},
        {"Name": "dogs", "Version": "2"},
    ]
    studio_output = tabulate(studio_rows, headers="keys")

    both_rows = [
        {"Name": "both", "Version": "1", "Studio": "\u2714", "Local": "\u2714"},
        {"Name": "cats", "Version": "1", "Studio": "\u2714", "Local": "\u2716"},
        {"Name": "dogs", "Version": "1", "Studio": "\u2714", "Local": "\u2716"},
        {"Name": "dogs", "Version": "2", "Studio": "\u2714", "Local": "\u2716"},
        {"Name": "local", "Version": "1", "Studio": "\u2716", "Local": "\u2714"},
    ]
    both_output = tabulate(both_rows, headers="keys")

    assert main(["datasets", "--local"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(local_output.splitlines())

    assert main(["datasets", "--studio"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(studio_output.splitlines())

    assert main(["datasets", "--local", "--studio"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(both_output.splitlines())

    assert main(["datasets", "--all"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(both_output.splitlines())

    assert main(["datasets"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(both_output.splitlines())
