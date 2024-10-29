def test_ls(benchmark, tmp_dir, bucket):
    benchmark("ls", bucket, "--anon")
