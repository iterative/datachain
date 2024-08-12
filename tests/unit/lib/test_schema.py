from datachain.lib.dc import C, DataChain
from datachain.lib.file import File


def test_column_filter_by_regex(test_session):
    names = [
        "hotdogs.txt",
        "dogs.txt",
        "dog.txt",
        "1dog.txt",
        "dogatxt.txt",
        "dog.txtx",
    ]

    dc = DataChain.from_values(file=[File(path=p) for p in names]).filter(
        C("file.path").regexp("dog\\.txt$")
    )

    assert set(dc.collect("file.path")) == {
        "dog.txt",
        "1dog.txt",
    }
