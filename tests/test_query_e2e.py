import os
import os.path
import signal
import subprocess
import sys
import textwrap
from io import TextIOWrapper
from textwrap import dedent
from threading import Thread
from typing import Callable

import pytest

tests_dir = os.path.dirname(os.path.abspath(__file__))

python_exc = sys.executable or "python3"


E2E_STEP_TIMEOUT_SEC = 60


E2E_STEPS = (
    {
        "command": (
            "datachain",
            "find",
            "--anon",
            "--name",
            "cat.1.*",
            "gs://dvcx-datalakes/dogs-and-cats/",
        ),
        "expected": dedent(
            """
            gs://dvcx-datalakes/dogs-and-cats/cat.1.jpg
            gs://dvcx-datalakes/dogs-and-cats/cat.1.json
            """
        ),
        "listing": True,
    },
    {
        "command": (
            python_exc,
            os.path.join(tests_dir, "scripts", "feature_class_parallel.py"),
        ),
        "expected_in": dedent(
            """
            dogs-and-cats/cat.1.jpg
            dogs-and-cats/cat.10.jpg
            dogs-and-cats/cat.100.jpg
            dogs-and-cats/cat.1000.jpg
            dogs-and-cats/cat.1001.jpg
            """
        ),
    },
    {
        "command": (
            python_exc,
            os.path.join(tests_dir, "scripts", "feature_class_parallel_data_model.py"),
        ),
        "expected_in": dedent(
            """
            dogs-and-cats/cat.1.jpg
            dogs-and-cats/cat.10.jpg
            dogs-and-cats/cat.100.jpg
            dogs-and-cats/cat.1000.jpg
            dogs-and-cats/cat.1001.jpg
            """
        ),
    },
    {
        # This reads from stdin, to emulate using the python REPL shell.
        "command": (python_exc, "-"),
        "stdin_file": os.path.join(
            tests_dir, "scripts", "feature_class_parallel_data_model.py"
        ),
        "expected_in": dedent(
            """
            dogs-and-cats/cat.1.jpg
            dogs-and-cats/cat.10.jpg
            dogs-and-cats/cat.100.jpg
            dogs-and-cats/cat.1000.jpg
            dogs-and-cats/cat.1001.jpg
            """
        ),
    },
    {
        "command": (
            "datachain",
            "query",
            os.path.join(tests_dir, "scripts", "feature_class.py"),
            "--columns",
            "file.path,emd.value",
        ),
        "expected_rows": dedent(
            """
                               file__path  emd__value
            0     dogs-and-cats/cat.1.jpg       512.0
            1    dogs-and-cats/cat.10.jpg       512.0
            2   dogs-and-cats/cat.100.jpg       512.0
            3  dogs-and-cats/cat.1000.jpg       512.0
            4  dogs-and-cats/cat.1001.jpg       512.0
            """
        ),
    },
    {
        "command": (
            python_exc,
            os.path.join(tests_dir, "scripts", "name_len_slow.py"),
        ),
        "interrupt_after": "UDF Processing Started",
        "expected_in_stderr": "KeyboardInterrupt",
        "expected_not_in_stderr": "semaphore",
    },
    {
        "command": ("datachain", "gc"),
        "expected": "Nothing to clean up.\n",
    },
)


def watch_process_thread(
    stream: TextIOWrapper, output_lines: list[str], watch_value: str, callback: Callable
) -> None:
    """
    Watches either the stdout or stderr stream from a given process,
    reads the output into output_lines, and watches for the given watch_value,
    then calls callback once found.
    """
    while (line := stream.readline()) != "":
        line = line.strip()
        output_lines.append(line)
        if watch_value in line:
            callback()


def communicate_and_interrupt_process(
    process: subprocess.Popen, interrupt_after: str
) -> tuple[str, str]:
    def interrupt_step() -> None:
        if sys.platform == "win32":
            # Windows has a different mechanism of sending a Ctrl-C event.
            process.send_signal(signal.CTRL_C_EVENT)
        else:
            process.send_signal(signal.SIGINT)

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    watch_threads = (
        Thread(
            target=watch_process_thread,
            name="Test-Query-E2E-Interrupt-stdout",
            daemon=True,
            args=[process.stdout, stdout_lines, interrupt_after, interrupt_step],
        ),
        Thread(
            target=watch_process_thread,
            name="Test-Query-E2E-Interrupt-stderr",
            daemon=True,
            args=[process.stderr, stderr_lines, interrupt_after, interrupt_step],
        ),
    )
    for t in watch_threads:
        t.start()
    process.wait(timeout=E2E_STEP_TIMEOUT_SEC)
    return "\n".join(stdout_lines), "\n".join(stderr_lines)


def run_step(step):  # noqa: PLR0912
    """Run an end-to-end query test step with a command and expected output."""
    command = step["command"]
    # Note that a process.returncode of -2 is the same as the shell returncode of 130
    # (canceled by KeyboardInterrupt)
    interrupt_exit_code = -2
    if sys.platform == "win32":
        # Windows has a different mechanism of creating a process group.
        popen_args = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        # This is STATUS_CONTROL_C_EXIT which is equivalent to 0xC000013A
        interrupt_exit_code = 3221225786
    else:
        popen_args = {"start_new_session": True}
    stdin_file = None
    if step.get("stdin_file"):
        # The "with" file open context manager cannot be used here without
        # additional code duplication, as a file is only opened if needed.
        stdin_file = open(step["stdin_file"])  # noqa: SIM115
    try:
        process = subprocess.Popen(  # noqa: S603
            command,
            shell=False,
            stdin=stdin_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            **popen_args,
        )
        interrupt_after = step.get("interrupt_after")
        if interrupt_after:
            stdout, stderr = communicate_and_interrupt_process(process, interrupt_after)
        else:
            stdout, stderr = process.communicate(timeout=E2E_STEP_TIMEOUT_SEC)
    finally:
        if stdin_file:
            stdin_file.close()

    if interrupt_after:
        if process.returncode not in (interrupt_exit_code, 1):
            print(f"Process stdout: {stdout}")
            print(f"Process stderr: {stderr}")
            raise RuntimeError(
                "Query script failed to interrupt correctly: "
                f"{process.returncode} Command: {command}"
            )
    elif process.returncode != 0:
        print(f"Process stdout: {stdout}")
        print(f"Process stderr: {stderr}")
        raise RuntimeError(
            "Query script failed with exit code: "
            f"{process.returncode} Command: {command}"
        )

    if step.get("sort_expected_lines"):
        assert sorted(stdout.split("\n")) == sorted(
            step["expected"].lstrip("\n").split("\n")
        )
    elif step.get("expected_in_stderr"):
        assert step["expected_in_stderr"] in stderr
        if step.get("expected_not_in_stderr"):
            assert step["expected_not_in_stderr"] not in stderr
    elif step.get("expected_in"):
        assert sorted(stdout.split("\n")) == sorted(
            step["expected_in"].lstrip("\n").split("\n")
        )
    elif step.get("expected_rows"):
        assert _comparable_row(stdout) == _comparable_row(step["expected_rows"])
    else:
        assert stdout == step["expected"].lstrip("\n")

    if step.get("listing"):
        assert "Listing" in stderr
    else:
        assert "Listing" not in stderr


@pytest.mark.e2e
def test_query_e2e(tmp_dir, catalog):
    """End-to-end CLI Query Test"""
    for step in E2E_STEPS:
        run_step(step)


def _comparable_row(output: str) -> str:
    return "\n".join(
        sorted(
            [_remove_serial_index(line) for line in output.lstrip("\n").splitlines()]
        )
    )


def _remove_serial_index(output: str) -> str:
    splits = textwrap.shorten(output, width=1000).strip().split(" ")
    if splits[0].isdigit():
        return " ".join(splits[1:])
    return " ".join(splits)
