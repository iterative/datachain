import os
import subprocess
import sys

import pytest

tests_dir = os.path.dirname(os.path.abspath(__file__))

python_exc = sys.executable or "python3"

E2E_STEP_TIMEOUT_SEC = 90


@pytest.mark.e2e
@pytest.mark.xdist_group(name="tmpfile")
def test_atomicity_feature_file(tmp_dir, catalog_tmpfile):
    command = (
        python_exc,
        os.path.join(tests_dir, "scripts", "feature_class_exception.py"),
    )
    if sys.platform == "win32":
        # Windows has a different mechanism of creating a process group.
        popen_args = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        # This is STATUS_CONTROL_C_EXIT which is equivalent to 0xC000013A
    else:
        popen_args = {"start_new_session": True}

    process = subprocess.Popen(  # noqa: S603
        command,
        shell=False,
        encoding="utf-8",
        env={
            **os.environ,
            "DATACHAIN__ID_GENERATOR": catalog_tmpfile.id_generator.serialize(),
            "DATACHAIN__METASTORE": catalog_tmpfile.metastore.serialize(),
            "DATACHAIN__WAREHOUSE": catalog_tmpfile.warehouse.serialize(),
        },
        **popen_args,
    )

    process.communicate(timeout=E2E_STEP_TIMEOUT_SEC)

    assert process.returncode == 1

    # No datasets should be left in the catalog
    assert list(catalog_tmpfile.list_datasets_versions()) == []
