import asyncio
import os
import sys
from typing import TYPE_CHECKING, Optional

import tabulate

from datachain.config import Config, ConfigLevel
from datachain.dataset import QUERY_DATASET_PREFIX, parse_dataset_name
from datachain.error import DataChainError
from datachain.remote.studio import StudioClient
from datachain.utils import STUDIO_URL

if TYPE_CHECKING:
    from argparse import Namespace

POST_LOGIN_MESSAGE = (
    "Once you've logged in, return here "
    "and you'll be ready to start using DataChain with Studio."
)


def process_jobs_args(args: "Namespace"):
    if args.cmd is None:
        print(
            f"Use 'datachain {args.command} --help' to see available options",
            file=sys.stderr,
        )
        return 1

    if args.cmd == "run":
        return create_job(
            args.file,
            args.team,
            args.env_file,
            args.env,
            args.workers,
            args.files,
            args.python_version,
            args.repository,
            args.req,
            args.req_file,
            args.priority,
            args.cluster,
        )

    if args.cmd == "cancel":
        return cancel_job(args.id, args.team)
    if args.cmd == "logs":
        return show_job_logs(args.id, args.team)

    if args.cmd == "ls":
        return list_jobs(args.status, args.team, args.limit)

    if args.cmd == "clusters":
        return list_clusters(args.team)

    raise DataChainError(f"Unknown command '{args.cmd}'.")


def process_auth_cli_args(args: "Namespace"):
    if args.cmd is None:
        print(
            f"Use 'datachain {args.command} --help' to see available options",
            file=sys.stderr,
        )
        return 1

    if args.cmd == "login":
        return login(args)
    if args.cmd == "logout":
        return logout(args.local)
    if args.cmd == "token":
        return token()
    if args.cmd == "team":
        return set_team(args)
    raise DataChainError(f"Unknown command '{args.cmd}'.")


def set_team(args: "Namespace"):
    if args.team_name is None:
        config = Config().read().get("studio", {})
        team = config.get("team")
        if team:
            print(f"Default team is '{team}'")
            return 0

        raise DataChainError(
            "No default team set. Use `datachain auth team <team_name>` to set one."
        )

    level = ConfigLevel.LOCAL if args.local else ConfigLevel.GLOBAL
    config = Config(level)
    with config.edit() as conf:
        studio_conf = conf.get("studio", {})
        studio_conf["team"] = args.team_name
        conf["studio"] = studio_conf

    print(f"Set default team to '{args.team_name}' in {config.config_file()}")


def login(args: "Namespace"):
    from dvc_studio_client.auth import StudioAuthError, get_access_token

    from datachain.remote.studio import get_studio_env_variable

    config = Config().read().get("studio", {})
    name = args.name
    hostname = (
        args.hostname
        or get_studio_env_variable("URL")
        or config.get("url")
        or STUDIO_URL
    )
    scopes = args.scopes

    if config.get("url", hostname) == hostname and "token" in config:
        raise DataChainError(
            "Token already exists. "
            "To login with a different token, "
            "logout using `datachain auth logout`."
        )

    open_browser = not args.no_open
    try:
        _, access_token = get_access_token(
            token_name=name,
            hostname=hostname,
            scopes=scopes,
            open_browser=open_browser,
            client_name="DataChain",
            post_login_message=POST_LOGIN_MESSAGE,
        )
    except StudioAuthError as exc:
        raise DataChainError(f"Failed to authenticate with Studio: {exc}") from exc

    level = ConfigLevel.LOCAL if args.local else ConfigLevel.GLOBAL
    config_path = save_config(hostname, access_token, level=level)
    print(f"Authentication complete. Saved token to {config_path}.")
    print("You can now use 'datachain auth team' to set the default team.")
    return 0


def logout(local: bool = False):
    level = ConfigLevel.LOCAL if local else ConfigLevel.GLOBAL
    with Config(level).edit() as conf:
        token = conf.get("studio", {}).get("token")
        if not token:
            raise DataChainError(
                "Not logged in to Studio. Log in with 'datachain auth login'."
            )

        del conf["studio"]["token"]

    print("Logged out from Studio. (you can log back in with 'datachain auth login')")


def token():
    config = Config().read().get("studio", {})
    token = config.get("token")
    if not token:
        raise DataChainError(
            "Not logged in to Studio. Log in with 'datachain auth login'."
        )

    print(token)


def list_datasets(team: Optional[str] = None, name: Optional[str] = None):
    def ds_full_name(ds: dict) -> str:
        return (
            f"{ds['project']['namespace']['name']}.{ds['project']['name']}.{ds['name']}"
        )

    if name:
        yield from list_dataset_versions(team, name)
        return

    client = StudioClient(team=team)

    response = client.ls_datasets()

    if not response.ok:
        raise DataChainError(response.message)

    if not response.data:
        return

    for d in response.data:
        name = d.get("name")
        full_name = ds_full_name(d)
        if name and name.startswith(QUERY_DATASET_PREFIX):
            continue

        for v in d.get("versions", []):
            version = v.get("version")
            yield (full_name, version)


def list_dataset_versions(team: Optional[str] = None, name: str = ""):
    client = StudioClient(team=team)

    namespace_name, project_name, name = parse_dataset_name(name)
    if not namespace_name or not project_name:
        raise DataChainError(f"Missing namespace or project form dataset name {name}")
    response = client.dataset_info(namespace_name, project_name, name)

    if not response.ok:
        raise DataChainError(response.message)

    if not response.data:
        return

    for v in response.data.get("versions", []):
        version = v.get("version")
        yield (name, version)


def edit_studio_dataset(
    team_name: Optional[str],
    name: str,
    namespace: str,
    project: str,
    new_name: Optional[str] = None,
    description: Optional[str] = None,
    attrs: Optional[list[str]] = None,
):
    client = StudioClient(team=team_name)
    response = client.edit_dataset(
        name, namespace, project, new_name, description, attrs
    )
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Dataset '{name}' updated in Studio")


def remove_studio_dataset(
    team_name: Optional[str],
    name: str,
    namespace: str,
    project: str,
    version: Optional[str] = None,
    force: Optional[bool] = False,
):
    client = StudioClient(team=team_name)
    response = client.rm_dataset(name, namespace, project, version, force)
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Dataset '{name}' removed from Studio")


def save_config(hostname, token, level=ConfigLevel.GLOBAL):
    config = Config(level)
    with config.edit() as conf:
        studio_conf = conf.get("studio", {})
        studio_conf["url"] = hostname
        studio_conf["token"] = token
        conf["studio"] = studio_conf

    return config.config_file()


def show_logs_from_client(client, job_id):
    # Sync usage
    async def _run():
        latest_status = None
        async for message in client.tail_job_logs(job_id):
            if "logs" in message:
                for log in message["logs"]:
                    print(log["message"], end="")
            elif "job" in message:
                latest_status = message["job"]["status"]
                print(f"\n>>>> Job is now in {latest_status} status.")
        return latest_status

    latest_status = asyncio.run(_run())

    response = client.dataset_job_versions(job_id)
    if not response.ok:
        raise DataChainError(response.message)

    response_data = response.data
    if response_data and response_data.get("dataset_versions"):
        dataset_versions = response_data.get("dataset_versions", [])
        print("\n\n>>>> Dataset versions created during the job:")
        for version in dataset_versions:
            print(f"    - {version.get('dataset_name')}@v{version.get('version')}")
    else:
        print("\n\nNo dataset versions created during the job.")

    exit_code_by_status = {
        "FAILED": 1,
        "CANCELLED": 2,
    }
    return exit_code_by_status.get(latest_status.upper(), 0) if latest_status else 0


def create_job(
    query_file: str,
    team_name: Optional[str],
    env_file: Optional[str] = None,
    env: Optional[list[str]] = None,
    workers: Optional[int] = None,
    files: Optional[list[str]] = None,
    python_version: Optional[str] = None,
    repository: Optional[str] = None,
    req: Optional[list[str]] = None,
    req_file: Optional[str] = None,
    priority: Optional[int] = None,
    cluster: Optional[str] = None,
):
    query_type = "PYTHON" if query_file.endswith(".py") else "SHELL"
    with open(query_file) as f:
        query = f.read()

    environment = "\n".join(env) if env else ""
    if env_file:
        with open(env_file) as f:
            environment = f.read() + "\n" + environment

    requirements = "\n".join(req) if req else ""
    if req_file:
        with open(req_file) as f:
            requirements = f.read() + "\n" + requirements

    client = StudioClient(team=team_name)
    file_ids = upload_files(client, files) if files else []

    response = client.create_job(
        query=query,
        query_type=query_type,
        environment=environment,
        workers=workers,
        query_name=os.path.basename(query_file),
        files=file_ids,
        python_version=python_version,
        repository=repository,
        requirements=requirements,
        priority=priority,
        cluster=cluster,
    )
    if not response.ok:
        raise DataChainError(response.message)

    if not response.data:
        raise DataChainError("Failed to create job")

    job_id = response.data.get("job", {}).get("id")
    print(f"Job {job_id} created")
    print("Open the job in Studio at", response.data.get("job", {}).get("url"))
    print("=" * 40)

    return show_logs_from_client(client, job_id)


def upload_files(client: StudioClient, files: list[str]) -> list[str]:
    file_ids = []
    for file in files:
        file_name = os.path.basename(file)
        with open(file, "rb") as f:
            file_content = f.read()
        response = client.upload_file(file_content, file_name)
        if not response.ok:
            raise DataChainError(response.message)

        if not response.data:
            raise DataChainError(f"Failed to upload file {file_name}")

        file_id = response.data.get("blob", {}).get("id")
        if file_id:
            file_ids.append(str(file_id))
    return file_ids


def cancel_job(job_id: str, team_name: Optional[str]):
    token = Config().read().get("studio", {}).get("token")
    if not token:
        raise DataChainError(
            "Not logged in to Studio. Log in with 'datachain auth login'."
        )

    client = StudioClient(team=team_name)
    response = client.cancel_job(job_id)
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Job {job_id} canceled")


def list_jobs(status: Optional[str], team_name: Optional[str], limit: int):
    client = StudioClient(team=team_name)
    response = client.get_jobs(status, limit)
    if not response.ok:
        raise DataChainError(response.message)

    jobs = response.data.get("jobs", [])
    if not jobs:
        print("No jobs found")
        return

    rows = [
        {
            "ID": job.get("id"),
            "Name": job.get("name"),
            "Status": job.get("status"),
            "Created at": job.get("created_at"),
            "Created by": job.get("created_by"),
        }
        for job in jobs
    ]

    print(tabulate.tabulate(rows, headers="keys", tablefmt="grid"))


def show_job_logs(job_id: str, team_name: Optional[str]):
    token = Config().read().get("studio", {}).get("token")
    if not token:
        raise DataChainError(
            "Not logged in to Studio. Log in with 'datachain auth login'."
        )

    client = StudioClient(team=team_name)
    return show_logs_from_client(client, job_id)


def list_clusters(team_name: Optional[str]):
    client = StudioClient(team=team_name)
    response = client.get_clusters()
    if not response.ok:
        raise DataChainError(response.message)

    clusters = response.data.get("clusters", [])
    if not clusters:
        print("No clusters found")
        return

    rows = [
        {
            "ID": cluster.get("id"),
            "Name": cluster.get("name"),
            "Status": cluster.get("status"),
            "Cloud Provider": cluster.get("cloud_provider"),
            "Cloud Credentials": cluster.get("cloud_credentials"),
            "Is Active": cluster.get("is_active"),
            "Max Workers": cluster.get("max_workers"),
        }
        for cluster in clusters
    ]

    print(tabulate.tabulate(rows, headers="keys", tablefmt="grid"))
