"""Automation using nox."""

import glob
import os

import nox

nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = "lint", "tests"
locations = "src", "tests"


@nox.session
def docs(session: nox.Session) -> None:
    session.install(".[docs]")
    session.run("mkdocs", "build")


@nox.session
def bench(session: nox.Session) -> None:
    session.install(".[tests]")
    session.run(
        "pytest",
        "-m",
        "benchmark",
        "--benchmark-group-by",
        "func",
        *session.posargs,
    )


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "pypy3.9", "pypy3.10"])
def tests(session: nox.Session) -> None:
    session.install(".[tests]")
    session.run(
        "pytest",
        "--cov",
        "--cov-config=pyproject.toml",
        "--cov-report=xml",
        "--durations=10",
        "--numprocesses=logical",
        *session.posargs,
        env={"COVERAGE_FILE": f".coverage.{session.python}"},
    )


@nox.session
def lint(session: nox.Session) -> None:
    session.install("pre-commit")
    session.install("-e", ".[dev,vector]")

    args = *(session.posargs or ("--show-diff-on-failure",)), "--all-files"
    session.run("pre-commit", "run", *args)


@nox.session
def build(session: nox.Session) -> None:
    session.install("build", "twine", "uv")
    session.run("python", "-m", "build", "--installer", "uv")
    dists = glob.glob("dist/*")
    session.run("twine", "check", *dists, silent=True)


@nox.session
def dev(session: nox.Session) -> None:
    """Sets up a python development environment for the project."""
    args = session.posargs or ("venv",)
    venv_dir = os.fsdecode(os.path.abspath(args[0]))

    session.log(f"Setting up virtual environment in {venv_dir}")
    session.install("virtualenv")
    session.run("virtualenv", venv_dir, silent=True)

    python = os.path.join(venv_dir, "bin/python")
    session.run(python, "-m", "pip", "install", "-e", ".[dev]", external=True)


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "pypy3.9", "pypy3.10"])
def examples(session: nox.Session) -> None:
    session.install(".[examples]")
    session.run(
        "pytest",
        "-m",
        "examples",
        *session.posargs,
    )
