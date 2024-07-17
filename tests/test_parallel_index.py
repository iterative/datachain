import os
import subprocess
import threading
import time
from textwrap import dedent

tests_dir = os.path.dirname(os.path.abspath(__file__))


def run_script(results, max_retries=5, delay=0.1):
    """
    Run a script step with retry logic for database locks.
    """
    command = ("python", os.path.join(tests_dir, "scripts", "parallel_index.py"))
    attempt = 0
    while attempt < max_retries:
        process = subprocess.run(  # noqa: S603
            command,
            shell=False,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
        if process.returncode == 0:
            results.append((process.stdout, process.returncode))
            break
        if "database is locked" in process.stderr:
            time.sleep(delay)  # Wait before retrying
            attempt += 1
        else:
            # Handle other errors immediately
            results.append((process.stdout, process.returncode))
            break
    else:
        # Max retries reached, handle accordingly
        results.append((process.stdout, process.returncode))


def test_parallel_index(tmp_dir, catalog):
    expected = dedent(
        """
        cat.1.jpg
        cat.10.jpg
        cat.100.jpg
        cat.1000.jpg
        cat.1001.jpg
        """
    )
    results = []
    _call_concurrently(*[lambda: run_script(results) for i in range(2)])
    assert len(results) == 2
    for output, returncode in results:
        assert returncode == 0
        assert expected.strip() in output.strip()


def _call_concurrently(*callables):
    """
    Run callables concurrently in separate threads.
    """
    threads = [threading.Thread(target=callable) for callable in callables]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
