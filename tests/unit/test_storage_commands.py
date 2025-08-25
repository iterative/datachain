import pytest

from datachain.cli import main
from datachain.utils import STUDIO_URL


@pytest.mark.parametrize(
    "command, recursive, team",
    [
        ("rm -s s3://my-bucket/data/content", False, None),
        ("rm -s s3://my-bucket/data/content --recursive", True, None),
        ("rm -s s3://my-bucket/data/content --team new_team", False, "new_team"),
        (
            "rm -s s3://my-bucket/data/content --team new_team --recursive",
            True,
            "new_team",
        ),
    ],
)
def test_rm_storage(requests_mock, capsys, studio_token, command, recursive, team):
    team_name = team or "team_name"  # default to team_name if not provided
    url = f"{STUDIO_URL}/api/datachain/storages/files?bucket=my-bucket&remote=s3"
    url += f"&recursive={recursive}&team={team_name}&paths=data/content"

    requests_mock.delete(
        url,
        json={"ok": True, "data": {"deleted": True}, "message": "", "status": 200},
        status_code=200,
    )

    result = main(command.split())
    assert result == 0
    out, _ = capsys.readouterr()
    assert "Deleted s3://my-bucket/data/content" in out

    assert requests_mock.called


@pytest.mark.parametrize(
    "command, recursive, team",
    [
        (
            "s3://my-bucket/data/content2",
            False,
            None,
        ),
        (
            "s3://my-bucket/data/content2 --recursive",
            True,
            None,
        ),
        (
            "s3://my-bucket/data/content2 --team new_team",
            False,
            "new_team",
        ),
        (
            "s3://my-bucket/data/content2 --team new_team --recursive",
            True,
            "new_team",
        ),
    ],
)
def test_mv_storage(requests_mock, capsys, studio_token, command, recursive, team):
    requests_mock.post(
        f"{STUDIO_URL}/api/datachain/storages/files/mv",
        json={"ok": True, "data": {"moved": True}, "message": "", "status": 200},
        status_code=200,
    )

    result = main(["mv", "-s", "s3://my-bucket/data/content", *command.split()])
    assert result == 0
    out, _ = capsys.readouterr()
    assert "Moved s3://my-bucket/data/content to s3://my-bucket/data/content2" in out

    assert requests_mock.called
    assert requests_mock.last_request.json() == {
        "bucket": "my-bucket",
        "newPath": "data/content2",
        "oldPath": "data/content",
        "recursive": recursive,
        "remote": "s3",
        "team": team or "team_name",
        "team_name": team or "team_name",
    }


def test_cp_storage_local_to_local(studio_token, tmp_dir):
    (tmp_dir / "path1").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "path1" / "file1.txt").write_text("file1")
    (tmp_dir / "path2").mkdir(parents=True, exist_ok=True)

    result = main(
        [
            "cp",
            "-s",
            str(tmp_dir / "path1" / "file1.txt"),
            str(tmp_dir / "path2" / "file1.txt"),
        ]
    )
    assert result == 0

    assert (tmp_dir / "path2" / "file1.txt").read_text() == "file1"


def test_cp_storage_local_to_s3(requests_mock, capsys, studio_token, tmp_dir):
    (tmp_dir / "path1").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "path1" / "file1.txt").write_text("file1")

    requests_mock.post(
        f"{STUDIO_URL}/api/datachain/storages/batch-presigned-urls",
        json={
            "urls": {
                "data/content": {
                    "url": "https://example.com/upload",
                    "fields": {"key": "value"},
                }
            },
            "headers": {},
            "method": "POST",
        },
    )
    requests_mock.post("https://example.com/upload", status_code=200)
    requests_mock.post(
        f"{STUDIO_URL}/api/datachain/storages/activity-logs",
        json={"success": True},
    )

    result = main(
        [
            "cp",
            "-s",
            str(tmp_dir / "path1" / "file1.txt"),
            "s3://my-bucket/data/content",
        ]
    )
    assert result == 0

    history = requests_mock.request_history
    assert len(history) == 3
    assert history[0].url == f"{STUDIO_URL}/api/datachain/storages/batch-presigned-urls"
    assert history[1].url == "https://example.com/upload"
    assert history[2].url == f"{STUDIO_URL}/api/datachain/storages/activity-logs"

    assert history[0].json() == {
        "bucket": "my-bucket",
        "paths": {"data/content": "text/plain"},
        "remote": "s3",
        "team": "team_name",
        "team_name": "team_name",
    }


def test_cp_remote_to_local(requests_mock, capsys, studio_token, tmp_dir):
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/storages/files/download?bucket=my-bucket&remote=s3&filepath=data%2Fcontent&team=team_name&team_name=team_name",
        json={
            "urls": {"data/content": "https://example.com/download"},
        },
    )
    requests_mock.get(
        "https://example.com/download",
        content=b"file1",
    )

    result = main(
        ["cp", "-s", "s3://my-bucket/data/content", str(tmp_dir / "file1.txt")]
    )
    assert result == 0
    assert (tmp_dir / "file1.txt").read_text() == "file1"

    history = requests_mock.request_history
    assert len(history) == 2
    assert history[1].url == "https://example.com/download"


def test_cp_s3_to_s3(requests_mock, capsys, studio_token, tmp_dir):
    requests_mock.post(
        f"{STUDIO_URL}/api/datachain/storages/files/cp",
        json={"copied": ["data/content"]},
        status_code=200,
    )

    result = main(
        ["cp", "-s", "s3://my-bucket/data/content", "s3://my-bucket/data/content2"]
    )
    assert result == 0

    history = requests_mock.request_history
    assert len(history) == 1
    assert history[0].url == f"{STUDIO_URL}/api/datachain/storages/files/cp"
    assert history[0].json() == {
        "bucket": "my-bucket",
        "newPath": "data/content2",
        "oldPath": "data/content",
        "recursive": False,
        "remote": "s3",
        "team": "team_name",
        "team_name": "team_name",
    }
