def test_version(benchmark):
    benchmark("--help", rounds=100)
