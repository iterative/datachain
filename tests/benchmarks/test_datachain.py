import pytest

from datachain.lib.dc import DataChain
from datachain.lib.webdataset_laion import process_laion_meta


@pytest.mark.benchmark
def test_datachain(tmp_dir, test_session, datasets, benchmark):
    def run_script(uri, **kwargs):
        DataChain.from_storage(uri, session=test_session, **kwargs).gen(
            emd=process_laion_meta
        ).map(
            stem=lambda file: file.get_file_stem(),
            params=["emd.file"],
            output=str,
        ).save("laion_emb")

    dataset = datasets / "laion-tiny.npz"
    assert dataset.is_file()
    benchmark(run_script, dataset.as_uri())
