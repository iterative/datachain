import pytest

from datachain.lib.dc import DataChain


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
def test_merge_union(cloud_test_catalog, inner, cloud_type):
    session = cloud_test_catalog.session

    src = cloud_test_catalog.src_uri

    dogs = DataChain.from_storage(f"{src}/dogs/*", session=session)
    cats = DataChain.from_storage(f"{src}/cats/*", session=session)

    dogs1 = dogs.map(sig1=lambda: 1, output={"sig1": int})
    dogs2 = dogs.map(sig2=lambda: 2, output={"sig2": int})
    cats1 = cats.map(sig1=lambda: 1, output={"sig1": int})

    merged = (dogs1 | cats1).merge(dogs2, "file.path", inner=inner)
    signals = merged.select("file.path", "sig1", "sig2").order_by("file.path").results()

    if inner:
        assert signals == [
            ("dogs/dog1", 1, 2),
            ("dogs/dog2", 1, 2),
            ("dogs/dog3", 1, 2),
            ("dogs/others/dog4", 1, 2),
        ]
    else:
        assert signals == [
            ("cats/cat1", 1, None),
            ("cats/cat2", 1, None),
            ("dogs/dog1", 1, 2),
            ("dogs/dog2", 1, 2),
            ("dogs/dog3", 1, 2),
            ("dogs/others/dog4", 1, 2),
        ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner1", [True, False])
@pytest.mark.parametrize("inner2", [True, False])
@pytest.mark.parametrize("inner3", [True, False])
def test_merge_multiple(cloud_test_catalog, inner1, inner2, inner3):
    session = cloud_test_catalog.session

    src = cloud_test_catalog.src_uri

    dogs = DataChain.from_storage(f"{src}/dogs/*", session=session)
    cats = DataChain.from_storage(f"{src}/cats/*", session=session)

    dogs_and_cats = dogs | cats
    dogs1 = dogs.map(sig1=lambda: 1, output={"sig1": int})
    cats1 = cats.map(sig2=lambda: 2, output={"sig2": int})
    dogs2 = dogs_and_cats.merge(dogs1, "file.path", inner=inner1)
    cats2 = dogs_and_cats.merge(cats1, "file.path", inner=inner2)
    merged = dogs2.merge(cats2, "file.path", inner=inner3)

    merged_signals = (
        merged.select("file.path", "sig1", "sig2").order_by("file.path").results()
    )

    if inner1 and inner2 and inner3:
        assert merged_signals == []
    elif inner1:
        assert merged_signals == [
            ("dogs/dog1", 1, None),
            ("dogs/dog2", 1, None),
            ("dogs/dog3", 1, None),
            ("dogs/others/dog4", 1, None),
        ]
    elif inner2 and inner3:
        assert merged_signals == [
            ("cats/cat1", None, 2),
            ("cats/cat2", None, 2),
        ]
    else:
        assert merged_signals == [
            ("cats/cat1", None, 2),
            ("cats/cat2", None, 2),
            ("dogs/dog1", 1, None),
            ("dogs/dog2", 1, None),
            ("dogs/dog3", 1, None),
            ("dogs/others/dog4", 1, None),
        ]
