import os
import shutil
from subprocess import check_output

import pytest
import virtualenv
from dulwich.porcelain import clone
from packaging import version


@pytest.fixture
def bucket():
    return "s3://noaa-bathymetry-pds/"


def pytest_generate_tests(metafunc):
    str_revs = metafunc.config.getoption("--datachain-revs")
    revs = str_revs.split(",") if str_revs else [None]
    if "datachain_rev" in metafunc.fixturenames:
        metafunc.parametrize("datachain_rev", revs, scope="session")


class VirtualEnv:
    def __init__(self, path) -> None:
        self.path = path
        self.bin = self.path / ("Scripts" if os.name == "nt" else "bin")

    def create(self) -> None:
        virtualenv.cli_run([os.fspath(self.path)])

    def run(self, cmd: str, *args: str, env=None) -> None:
        exe = self.which(cmd)
        check_output([exe, *args], env=env)  # noqa: S603

    def which(self, cmd: str) -> str:
        assert self.bin.exists()
        return shutil.which(cmd, path=self.bin) or cmd


@pytest.fixture(scope="session", name="make_datachain_venv")
def fixture_make_datachain_venv(tmp_path_factory):
    def _make_datachain_venv(name):
        venv_dir = tmp_path_factory.mktemp(f"datachain-venv-{name}")
        venv = VirtualEnv(venv_dir)
        venv.create()
        return venv

    return _make_datachain_venv


@pytest.fixture(scope="session", name="datachain_venvs")
def fixture_datachain_venvs():
    return {}


@pytest.fixture(scope="session", name="datachain_git_repo")
def fixture_datachain_git_repo(tmp_path_factory, test_config):
    url = test_config.datachain_git_repo

    if os.path.isdir(url):
        return url

    tmp_path = os.fspath(tmp_path_factory.mktemp("datachain-git-repo"))
    clone(url, tmp_path)

    return tmp_path


@pytest.fixture(scope="session", name="datachain_bin")
def fixture_datachain_bin(
    datachain_rev,
    datachain_venvs,
    make_datachain_venv,
    datachain_git_repo,
    test_config,
):
    if datachain_rev:
        venv = datachain_venvs.get(datachain_rev)
        if not venv:
            venv = make_datachain_venv(datachain_rev)
            venv.run("pip", "install", "-U", "pip")
            venv.run(
                "pip", "install", f"git+file://{datachain_git_repo}@{datachain_rev}"
            )
            datachain_venvs[datachain_rev] = venv
        datachain_bin = venv.which("datachain")
    else:
        datachain_bin = test_config.datachain_bin

    def _datachain_bin(*args):
        return check_output([datachain_bin, *args], text=True)  # noqa: S603

    actual = version.parse(_datachain_bin("--version"))
    _datachain_bin.version = (actual.major, actual.minor, actual.micro)

    return _datachain_bin


@pytest.fixture(scope="function", name="make_bench")
def fixture_make_bench(request):
    def _make_bench(name):
        import pytest_benchmark.plugin

        # hack from https://github.com/ionelmc/pytest-benchmark/issues/166
        bench = pytest_benchmark.plugin.benchmark.__pytest_wrapped__.obj(request)

        suffix = f"-{name}"

        def add_suffix(_name):
            start, sep, end = _name.partition("[")
            return start + suffix + sep + end

        bench.name = add_suffix(bench.name)
        bench.fullname = add_suffix(bench.fullname)

        return bench

    return _make_bench


@pytest.fixture(
    scope="function", params=[pytest.param(None, marks=pytest.mark.benchmark)]
)
def bench_datachain(datachain_bin, make_bench):
    def _bench_datachain(*args, **kwargs):
        name = kwargs.pop("name", None)
        name = f"-{name}" if name else ""
        bench = make_bench(args[0] + name)
        return bench.pedantic(datachain_bin, args=args, **kwargs)

    return _bench_datachain
