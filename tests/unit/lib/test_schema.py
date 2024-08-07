from datachain.lib.dc import C, DataChain
from datachain.lib.file import File


def test_column_filter_by_regex():
    names = [
        "hotdogs.txt",
        "dogs.txt",
        "dog.txt",
        "1dog.txt",
        "dogatxt.txt",
        "dog.txtx",
    ]

    dc = DataChain.from_values(file=[File(name=name) for name in names]).filter(
        C("file.name").regexp("dog\\.txt$")
    )

    assert set(dc.collect("file.name")) == {
        "dog.txt",
        "1dog.txt",
    }
