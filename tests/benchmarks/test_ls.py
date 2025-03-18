from datachain.cli import ls


def test_ls(benchmark, tmp_dir):
    bucket = "s3://noaa-bathymetry-pds/"
    benchmark.pedantic(ls, args=([bucket],), kwargs={"client_config": {"anon": True}})
