def test_ls(bench_datachain, tmp_dir, bucket):
    bench_datachain("ls", bucket, "--anon")
