from datachain.lib.dc import DataChain
from datachain.lib.webdataset_laion import process_laion_meta


def test_datachain(tmp_dir, test_session, datasets, benchmark):
    def run_script(uri, **kwargs):
        DataChain.from_storage(uri, session=test_session, **kwargs).gen(
            emd=process_laion_meta
        ).settings(
            # Disable `prefetch` for `map()` because `process_laion_meta` repeatedly
            # returns the dataset file. This causes `prefetch` to download and
            # remove the file multiple times unnecessarily, slowing down the process.
            prefetch=0,
        ).map(
            stem=lambda file: file.get_file_stem(),
            params=["emd.file"],
            output=str,
        ).save("laion_emb")

    dataset = datasets / "laion-tiny.npz"
    assert dataset.is_file()
    benchmark(run_script, dataset.as_uri())
