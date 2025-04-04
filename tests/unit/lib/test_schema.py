import datachain as dc
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

    chain = dc.read_values(file=[File(path=p) for p in names]).filter(
        dc.C("file.path").regexp("dog\\.txt$")
    )

    assert set(chain.collect("file.path")) == {
        "dog.txt",
        "1dog.txt",
    }
