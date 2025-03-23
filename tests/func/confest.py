import os
import signal
import subprocess  # nosec B404
from time import sleep

import pytest

WORKER_SHUTDOWN_WAIT_SEC = 30


@pytest.fixture()
def run_datachain_worker():
    if not os.environ.get("DATACHAIN_DISTRIBUTED"):
        pytest.skip("Distributed tests are disabled")
    # This worker can take several tasks in parallel, as it's very handy
    # for testing, where we don't want [yet] to constrain the number of
    # available workers.
    workers = []
    worker_cmd = [
        "celery",
        "-A",
        "datachain_server.distributed",
        "worker",
        "--loglevel=INFO",
        "-Q",
        "datachain-worker",
        "-n",
        "datachain-worker-tests",
    ]
    workers.append(subprocess.Popen(worker_cmd, shell=False))  # noqa: S603
    try:
        from datachain_server.distributed import app

        inspect = app.control.inspect()
        attempts = 0
        # Wait 10 seconds for the Celery worker(s) to be up
        while not inspect.active() and attempts < 10:
            sleep(1)
            attempts += 1

        if attempts == 10:
            raise RuntimeError("Celery worker(s) did not start in time")

        yield workers
    finally:
        for worker in workers:
            os.kill(worker.pid, signal.SIGTERM)
        for worker in workers:
            try:
                worker.wait(timeout=WORKER_SHUTDOWN_WAIT_SEC)
            except subprocess.TimeoutExpired:
                os.kill(worker.pid, signal.SIGKILL)
