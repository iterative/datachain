import shutil
import subprocess


def test_version(benchmark):
    bin = shutil.which("datachain")
    benchmark(subprocess.check_call, [bin, "--help"])
