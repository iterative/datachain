import io
import json
import os
from contextlib import suppress
from pathlib import Path
from textwrap import dedent
from urllib.parse import urlparse

import pytest
import yaml
from fsspec.implementations.local import LocalFileSystem

from datachain.catalog import parse_edatachain_file
from datachain.cli import garbage_collect
from datachain.error import (
    QueryScriptCompileError,
    QueryScriptDatasetNotFound,
    QueryScriptRunError,
    StorageNotFoundError,
)
from tests.data import ENTRIES
from tests.utils import (
    DEFAULT_TREE,
    TARRED_TREE,
    assert_row_names,
    create_tar_dataset,
    make_index,
    skip_if_not_sqlite,
    tree_from_path,
)


@pytest.fixture
def pre_created_ds_name():
    return "pre_created_dataset"


@pytest.fixture
def mock_os_pipe(mocker):
    r, w = os.pipe()
    mocker.patch("os.pipe", return_value=(r, w))

    try:
        yield (r, w)
    finally:
        with suppress(OSError):
            os.close(r)
        with suppress(OSError):
            os.close(w)


@pytest.fixture
def mock_popen(mocker):
    m = mocker.patch(
        "subprocess.Popen", returncode=0, stdout=io.StringIO(), stderr=io.StringIO()
    )
    m.return_value.__enter__.return_value = m
    # keep in sync with the returncode
    m.poll.side_effect = lambda: m.returncode if m.poll.call_count > 1 else None
    return m


@pytest.fixture
def mock_popen_dataset_created(
    mock_popen, cloud_test_catalog, mock_os_pipe, listed_bucket
):
    # create dataset which would be created in subprocess
    ds_name = cloud_test_catalog.catalog.generate_query_dataset_name()
    ds_version = 1
    cloud_test_catalog.catalog.create_dataset_from_sources(
        ds_name,
        [f"{cloud_test_catalog.src_uri}/dogs/*"],
        recursive=True,
    )

    _, w = mock_os_pipe
    with open(w, mode="w", closefd=False) as f:
        f.write(json.dumps({"dataset": (ds_name, ds_version)}))

    mock_popen.configure_mock(stdout=io.StringIO("user log 1\nuser log 2"))
    yield mock_popen


@pytest.fixture
def fake_index(catalog):
    src = "s3://whatever"
    make_index(catalog, src, ENTRIES)
    return src


def test_find(catalog, fake_index):
    src_uri = fake_index
    dirs = ["cats/", "dogs/", "dogs/others/"]
    expected_paths = dirs + [entry.full_path for entry in ENTRIES]
    assert set(catalog.find([src_uri])) == {
        f"{src_uri}/{path}" for path in expected_paths
    }

    with pytest.raises(FileNotFoundError):
        set(catalog.find([f"{src_uri}/does_not_exist"]))


def test_find_names_paths_size_type(catalog, fake_index):
    src_uri = fake_index

    assert set(catalog.find([src_uri], names=["*cat*"])) == {
        f"{src_uri}/cats/",
        f"{src_uri}/cats/cat1",
        f"{src_uri}/cats/cat2",
    }

    assert set(catalog.find([src_uri], names=["*cat*"], typ="dir")) == {
        f"{src_uri}/cats/",
    }

    assert len(list(catalog.find([src_uri], names=["*CAT*"]))) == 0

    assert set(catalog.find([src_uri], inames=["*CAT*"])) == {
        f"{src_uri}/cats/",
        f"{src_uri}/cats/cat1",
        f"{src_uri}/cats/cat2",
    }

    assert set(catalog.find([src_uri], paths=["*cats/cat*"])) == {
        f"{src_uri}/cats/cat1",
        f"{src_uri}/cats/cat2",
    }

    assert len(list(catalog.find([src_uri], paths=["*caTS/CaT**"]))) == 0

    assert set(catalog.find([src_uri], ipaths=["*caTS/CaT*"])) == {
        f"{src_uri}/cats/cat1",
        f"{src_uri}/cats/cat2",
    }

    assert set(catalog.find([src_uri], size="5", typ="f")) == {
        f"{src_uri}/description",
    }

    assert set(catalog.find([src_uri], size="-3", typ="f")) == {
        f"{src_uri}/dogs/dog2",
    }


def test_find_names_columns(cloud_test_catalog, cloud_type):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    owner = "webfile" if cloud_type == "s3" else ""

    src_uri_path = src_uri
    if cloud_type == "file":
        src_uri_path = LocalFileSystem._strip_protocol(src_uri)

    assert set(
        catalog.find(
            [src_uri],
            names=["*cat*"],
            columns=["du", "name", "owner", "path", "size", "type"],
        )
    ) == {
        "\t".join(columns)
        for columns in [
            ["8", "cats", "", f"{src_uri_path}/cats/", "0", "d"],
            ["4", "cat1", owner, f"{src_uri_path}/cats/cat1", "4", "f"],
            ["4", "cat2", owner, f"{src_uri_path}/cats/cat2", "4", "f"],
        ]
    }


@pytest.mark.parametrize(
    "recursive,star,dir_exists",
    (
        (True, True, False),
        (True, False, False),
        (True, False, True),
        (False, True, False),
        (False, False, False),
    ),
)
def test_cp_root(cloud_test_catalog, recursive, star, dir_exists, cloud_type):
    src_uri = cloud_test_catalog.src_uri
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog

    dest = working_dir / "data"

    if star:
        src_path = f"{src_uri}/*"
    else:
        src_path = src_uri
        if cloud_type == "file":
            src_path += "/"

    if star:
        with pytest.raises(FileNotFoundError):
            catalog.cp([src_path], str(dest), recursive=recursive)

    if dir_exists or star:
        dest.mkdir()

    catalog.cp([src_path], str(dest), recursive=recursive)

    if not star and not recursive:
        # The root directory is skipped, so nothing is copied
        assert tree_from_path(dest) == {}
        return

    assert (dest / "description").read_text() == "Cats and Dogs"

    # Testing DataChain File Contents
    assert dest.with_suffix(".edatachain").is_file()
    edatachain_contents = yaml.safe_load(dest.with_suffix(".edatachain").read_text())
    assert len(edatachain_contents) == 1
    data = edatachain_contents[0]
    assert data["data-source"]["uri"] == src_path.rstrip("/")
    expected_file_count = 7 if recursive else 1
    assert len(data["files"]) == expected_file_count
    files_by_name = {f["name"]: f for f in data["files"]}

    # Directories should never be saved
    assert "cats" not in files_by_name
    assert "dogs" not in files_by_name
    assert "others" not in files_by_name
    assert "dogs/others" not in files_by_name

    # Description is always copied (if anything is copied)
    prefix = (
        "" if star or (recursive and not dir_exists) or cloud_type == "file" else "/"
    )
    assert files_by_name[f"{prefix}description"]["size"] == 13

    if recursive:
        assert tree_from_path(dest) == DEFAULT_TREE
        assert files_by_name[f"{prefix}cats/cat1"]["size"] == 4
        assert files_by_name[f"{prefix}cats/cat2"]["size"] == 4
        assert files_by_name[f"{prefix}dogs/dog1"]["size"] == 4
        assert files_by_name[f"{prefix}dogs/dog2"]["size"] == 3
        assert files_by_name[f"{prefix}dogs/dog3"]["size"] == 4
        assert files_by_name[f"{prefix}dogs/others/dog4"]["size"] == 4
        return

    assert (dest / "cats").exists() is False
    assert (dest / "dogs").exists() is False
    for prefix in ["/", ""]:
        assert f"{prefix}cats/cat1" not in files_by_name
        assert f"{prefix}cats/cat2" not in files_by_name
        assert f"{prefix}dogs/dog1" not in files_by_name
        assert f"{prefix}dogs/dog2" not in files_by_name
        assert f"{prefix}dogs/dog3" not in files_by_name
        assert f"{prefix}dogs/others/dog4" not in files_by_name


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "gs", "azure"],
    indirect=True,
)
def test_cp_local_dataset(cloud_test_catalog, dogs_dataset):
    skip_if_not_sqlite()
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog

    dest = working_dir / "data"
    dest.mkdir()

    dataset_uri = dogs_dataset.uri(version=1)

    catalog.cp([dataset_uri], str(dest))

    parsed = urlparse(str(cloud_test_catalog.src))
    netloc = Path(parsed.netloc.strip("/"))
    path = Path(parsed.path.strip("/"))

    assert tree_from_path(dest / netloc / path) == {
        "dogs": {
            "dog1": "woof",
            "dog2": "arf",
            "dog3": "bark",
            "others": {"dog4": "ruff"},
        }
    }


@pytest.mark.parametrize("tree", [TARRED_TREE], indirect=True)
@pytest.mark.parametrize("suffix", ["/", "/*"])
@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("dir_exists", [False, True])
@pytest.mark.xfail(reason="Missing support for v-objects in cp")
def test_cp_tar_root(cloud_test_catalog, suffix, recursive, dir_exists):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    create_tar_dataset(catalog, ctc.src_uri, "tarred")
    dest = ctc.working_dir / "data"
    if dir_exists:
        dest.mkdir()
    src = f"ds://tarred/animals.tar{suffix}"
    dest_path = str(dest) + "/"

    if not dir_exists and suffix == "/*":
        with pytest.raises(FileNotFoundError):
            catalog.cp([src], dest_path, recursive=recursive, no_edatachain_file=True)
        return

    catalog.cp([src], dest_path, recursive=recursive, no_edatachain_file=True)

    expected = DEFAULT_TREE.copy()
    if not recursive:
        # Directories are not copied
        if suffix == "/":
            expected = {}
        else:
            for key in list(expected):
                if isinstance(expected[key], dict):
                    del expected[key]

    assert tree_from_path(dest) == expected


@pytest.mark.parametrize("tree", [TARRED_TREE], indirect=True)
@pytest.mark.xfail(reason="Missing support for v-objects in cp")
def test_cp_full_tar(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    create_tar_dataset(catalog, ctc.src_uri, "tarred")
    dest = ctc.working_dir / "data"
    dest.mkdir()
    src = "ds://tarred/"
    catalog.cp([src], str(dest), recursive=True, no_edatachain_file=True)

    assert tree_from_path(dest, binary=True) == TARRED_TREE


@pytest.mark.parametrize(
    "recursive,star,slash,dir_exists",
    (
        (True, True, False, False),
        (True, False, False, False),
        (True, False, False, True),
        (True, False, True, False),
        (False, True, False, False),
        (False, False, False, False),
        (False, False, True, False),
    ),
)
def test_cp_subdir(cloud_test_catalog, recursive, star, slash, dir_exists):
    src_uri = f"{cloud_test_catalog.src_uri}/dogs"
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog

    dest = working_dir / "data"

    if star:
        src_path = f"{src_uri}/*"
    elif slash:
        src_path = f"{src_uri}/"
    else:
        src_path = src_uri

    if star:
        with pytest.raises(FileNotFoundError):
            catalog.cp([src_path], str(dest), recursive=recursive)

    if dir_exists or star:
        dest.mkdir()

    catalog.cp([src_path], str(dest), recursive=recursive)

    if not star and not recursive:
        # Directories are skipped, so nothing is copied
        assert tree_from_path(dest) == {}
        return

    # Testing DataChain File Contents
    assert dest.with_suffix(".edatachain").is_file()
    edatachain_contents = yaml.safe_load(dest.with_suffix(".edatachain").read_text())
    assert len(edatachain_contents) == 1
    data = edatachain_contents[0]
    assert data["data-source"]["uri"] == src_path.rstrip("/")
    expected_file_count = 4 if recursive else 3
    assert len(data["files"]) == expected_file_count
    files_by_name = {f["name"]: f for f in data["files"]}

    # Directories should never be saved
    assert "others" not in files_by_name
    assert "dogs/others" not in files_by_name

    if not dir_exists:
        assert (dest / "dog1").read_text() == "woof"
        assert (dest / "dog2").read_text() == "arf"
        assert (dest / "dog3").read_text() == "bark"
        assert (dest / "dogs").exists() is False
        assert files_by_name["dog1"]["size"] == 4
        assert files_by_name["dog2"]["size"] == 3
        assert files_by_name["dog3"]["size"] == 4
        if recursive:
            assert (dest / "others" / "dog4").read_text() == "ruff"
            assert files_by_name["others/dog4"]["size"] == 4
        else:
            assert (dest / "others").exists() is False
            assert "others/dog4" not in files_by_name
        return

    assert tree_from_path(dest / "dogs") == DEFAULT_TREE["dogs"]
    assert (dest / "dog1").exists() is False
    assert (dest / "dog2").exists() is False
    assert (dest / "dog3").exists() is False
    assert (dest / "others").exists() is False
    assert files_by_name["dogs/dog1"]["size"] == 4
    assert files_by_name["dogs/dog2"]["size"] == 3
    assert files_by_name["dogs/dog3"]["size"] == 4
    assert files_by_name["dogs/others/dog4"]["size"] == 4


@pytest.mark.parametrize("tree", [TARRED_TREE], indirect=True)
@pytest.mark.parametrize("path", ["*/dogs", "animals.tar/dogs"])
@pytest.mark.parametrize("suffix", ["", "/", "/*"])
@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("dir_exists", [False, True])
@pytest.mark.xfail(reason="Missing support for v-objects in cp")
def test_cp_tar_subdir(cloud_test_catalog, path, suffix, recursive, dir_exists):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    create_tar_dataset(catalog, ctc.src_uri, "tarred")
    dest = ctc.working_dir / "data"
    if dir_exists:
        dest.mkdir()
    src = f"ds://tarred/{path}{suffix}"

    if not dir_exists and suffix == "/*":
        with pytest.raises(FileNotFoundError):
            catalog.cp([src], str(dest), recursive=recursive)
        return

    catalog.cp([src], str(dest), recursive=recursive)

    expected = DEFAULT_TREE["dogs"].copy()
    if suffix in ("",) and dir_exists:
        expected = {"dogs": expected}
    if not recursive:
        # Directories are not copied
        if not dir_exists or suffix == "/":
            expected = {}
        else:
            for key in list(expected):
                if isinstance(expected[key], dict):
                    del expected[key]

    assert tree_from_path(dest) == expected


@pytest.mark.parametrize(
    "recursive,star,slash",
    (
        (True, True, False),
        (True, False, False),
        (True, False, True),
        (False, True, False),
        (False, False, False),
        (False, False, True),
    ),
)
def test_cp_multi_subdir(cloud_test_catalog, recursive, star, slash):  # noqa: PLR0915
    sources = [
        f"{cloud_test_catalog.src_uri}/cats",
        f"{cloud_test_catalog.src_uri}/dogs",
    ]
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog

    dest = working_dir / "data"

    if star:
        src_paths = [f"{src}/*" for src in sources]
    elif slash:
        src_paths = [f"{src}/" for src in sources]
    else:
        src_paths = sources

    with pytest.raises(FileNotFoundError):
        catalog.cp(src_paths, str(dest), recursive=recursive)

    dest.mkdir()

    catalog.cp(src_paths, str(dest), recursive=recursive)

    if not star and not recursive:
        # Directories are skipped, so nothing is copied
        assert tree_from_path(dest) == {}
        return

    # Testing DataChain File Contents
    assert dest.with_suffix(".edatachain").is_file()
    edatachain_contents = yaml.safe_load(dest.with_suffix(".edatachain").read_text())
    assert len(edatachain_contents) == 2
    data_cats = edatachain_contents[0]
    data_dogs = edatachain_contents[1]
    assert data_cats["data-source"]["uri"] == src_paths[0].rstrip("/")
    assert data_dogs["data-source"]["uri"] == src_paths[1].rstrip("/")
    assert len(data_cats["files"]) == 2
    assert len(data_dogs["files"]) == 4 if recursive else 3
    cat_files_by_name = {f["name"]: f for f in data_cats["files"]}
    dog_files_by_name = {f["name"]: f for f in data_dogs["files"]}

    # Directories should never be saved
    assert "others" not in dog_files_by_name
    assert "dogs/others" not in dog_files_by_name

    if star or slash:
        assert (dest / "cat1").read_text() == "meow"
        assert (dest / "cat2").read_text() == "mrow"
        assert (dest / "dog1").read_text() == "woof"
        assert (dest / "dog2").read_text() == "arf"
        assert (dest / "dog3").read_text() == "bark"
        assert (dest / "cats").exists() is False
        assert (dest / "dogs").exists() is False
        assert cat_files_by_name["cat1"]["size"] == 4
        assert cat_files_by_name["cat2"]["size"] == 4
        assert dog_files_by_name["dog1"]["size"] == 4
        assert dog_files_by_name["dog2"]["size"] == 3
        assert dog_files_by_name["dog3"]["size"] == 4
        if recursive:
            assert (dest / "others" / "dog4").read_text() == "ruff"
            assert dog_files_by_name["others/dog4"]["size"] == 4
        else:
            assert (dest / "others").exists() is False
            assert "others/dog4" not in dog_files_by_name
        return

    assert (dest / "cats" / "cat1").read_text() == "meow"
    assert (dest / "cats" / "cat2").read_text() == "mrow"
    assert (dest / "dogs" / "dog1").read_text() == "woof"
    assert (dest / "dogs" / "dog2").read_text() == "arf"
    assert (dest / "dogs" / "dog3").read_text() == "bark"
    assert (dest / "dogs" / "others" / "dog4").read_text() == "ruff"
    assert (dest / "cat1").exists() is False
    assert (dest / "cat2").exists() is False
    assert (dest / "dog1").exists() is False
    assert (dest / "dog2").exists() is False
    assert (dest / "dog3").exists() is False
    assert (dest / "others").exists() is False
    assert cat_files_by_name["cats/cat1"]["size"] == 4
    assert cat_files_by_name["cats/cat2"]["size"] == 4
    assert dog_files_by_name["dogs/dog1"]["size"] == 4
    assert dog_files_by_name["dogs/dog2"]["size"] == 3
    assert dog_files_by_name["dogs/dog3"]["size"] == 4
    assert dog_files_by_name["dogs/others/dog4"]["size"] == 4


def test_cp_double_subdir(cloud_test_catalog):
    src_path = f"{cloud_test_catalog.src_uri}/dogs/others"
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog
    dest = working_dir / "data"

    catalog.cp([src_path], str(dest), recursive=True)

    # Testing DataChain File Contents
    assert dest.with_suffix(".edatachain").is_file()
    edatachain_contents = yaml.safe_load(dest.with_suffix(".edatachain").read_text())
    assert len(edatachain_contents) == 1
    data = edatachain_contents[0]
    assert data["data-source"]["uri"] == src_path.rstrip("/")
    assert len(data["files"]) == 1
    files_by_name = {f["name"]: f for f in data["files"]}

    # Directories should never be saved
    assert "others" not in files_by_name
    assert "dogs/others" not in files_by_name

    assert (dest / "dogs").exists() is False
    assert (dest / "others").exists() is False
    assert (dest / "dog4").read_text() == "ruff"
    assert files_by_name["dog4"]["size"] == 4


@pytest.mark.parametrize("no_glob", (True, False))
def test_cp_single_file(cloud_test_catalog, no_glob):
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog
    dest = working_dir / "data"
    src_path = f"{cloud_test_catalog.src_uri}/dogs/dog1"
    dest.mkdir()

    catalog.cp(
        [src_path], str(dest / "local_dog"), no_edatachain_file=True, no_glob=no_glob
    )

    assert tree_from_path(dest) == {"local_dog": "woof"}


@pytest.mark.parametrize("tree", [{"foo": "original"}], indirect=True)
def test_storage_mutation(cloud_test_catalog):
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog
    src_path = f"{cloud_test_catalog.src_uri}/foo"

    dest = working_dir / "data1"
    dest.mkdir()
    catalog.cp([src_path], str(dest / "local"), no_edatachain_file=True)
    assert tree_from_path(dest) == {"local": "original"}

    # Storage modified without reindexing, we get the old version from cache.
    (cloud_test_catalog.src / "foo").write_text("modified")
    dest = working_dir / "data2"
    dest.mkdir()
    catalog.cp([src_path], str(dest / "local"), no_edatachain_file=True)
    assert tree_from_path(dest) == {"local": "original"}

    # Storage modified without reindexing.
    # Since the old version cannot be found in storage or cache, it's an error.
    catalog.cache.clear()
    dest = working_dir / "data3"
    dest.mkdir()
    with pytest.raises(FileNotFoundError):
        catalog.cp([src_path], str(dest / "local"), no_edatachain_file=True)
    assert tree_from_path(dest) == {}

    # Storage modified with reindexing, we get the new version.
    catalog.index([cloud_test_catalog.src_uri], update=True)
    dest = working_dir / "data4"
    dest.mkdir()
    catalog.cp([src_path], str(dest / "local"), no_edatachain_file=True)
    assert tree_from_path(dest) == {"local": "modified"}


def test_cp_edatachain_file_options(cloud_test_catalog):
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog
    dest = working_dir / "data"
    src_path = f"{cloud_test_catalog.src_uri}/dogs/*"
    edatachain_file = working_dir / "custom_name.edatachain"

    catalog.cp(
        [src_path],
        str(dest),
        recursive=False,
        edatachain_only=True,
        edatachain_file=str(edatachain_file),
    )

    assert (dest / "dog1").exists() is False
    assert (dest / "dog2").exists() is False
    assert (dest / "dog3").exists() is False
    assert (dest / "dogs").exists() is False
    assert (dest / "others").exists() is False
    assert dest.with_suffix(".edatachain").exists() is False

    # Testing DataChain File Contents
    assert edatachain_file.is_file()
    edatachain_contents = yaml.safe_load(edatachain_file.read_text())
    assert len(edatachain_contents) == 1
    data = edatachain_contents[0]
    assert data["data-source"]["uri"] == src_path
    expected_file_count = 3
    assert len(data["files"]) == expected_file_count
    files_by_name = {f["name"]: f for f in data["files"]}

    assert parse_edatachain_file(str(edatachain_file)) == edatachain_contents

    # Directories should never be saved
    assert "others" not in files_by_name
    assert "dogs/others" not in files_by_name

    assert files_by_name["dog1"]["size"] == 4
    assert files_by_name["dog2"]["size"] == 3
    assert files_by_name["dog3"]["size"] == 4
    assert "others/dog4" not in files_by_name

    with pytest.raises(FileNotFoundError):
        # Should fail, as * will not be expanded
        catalog.cp(
            [src_path],
            str(dest),
            recursive=False,
            edatachain_only=True,
            edatachain_file=str(edatachain_file),
            no_glob=True,
        )

    # Should succeed, as the DataChain file exists check will be skipped
    edatachain_only_data = catalog.cp(
        [src_path],
        str(dest),
        recursive=False,
        edatachain_only=True,
        edatachain_file=str(edatachain_file),
        force=True,
    )

    # Check the returned DataChain data contents
    assert len(edatachain_only_data) == len(edatachain_contents)
    edatachain_only_source = edatachain_only_data[0]
    assert edatachain_only_source["data-source"]["uri"] == src_path.rstrip("/")
    assert edatachain_only_source["files"] == data["files"]


def test_cp_edatachain_file_sources(cloud_test_catalog):  # noqa: PLR0915
    sources = [
        f"{cloud_test_catalog.src_uri}/cats/",
        f"{cloud_test_catalog.src_uri}/dogs/*",
    ]
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog

    dest = working_dir / "data"

    edatachain_files = [
        working_dir / "custom_cats.edatachain",
        working_dir / "custom_dogs.edatachain",
    ]

    catalog.cp(
        sources[:1],
        str(dest),
        recursive=True,
        edatachain_only=True,
        edatachain_file=str(edatachain_files[0]),
    )

    catalog.cp(
        sources[1:],
        str(dest),
        recursive=True,
        edatachain_only=True,
        edatachain_file=str(edatachain_files[1]),
    )

    # Files should not be copied yet
    assert (dest / "cat1").exists() is False
    assert (dest / "cat2").exists() is False
    assert (dest / "cats").exists() is False
    assert (dest / "dog1").exists() is False
    assert (dest / "dog2").exists() is False
    assert (dest / "dog3").exists() is False
    assert (dest / "dogs").exists() is False
    assert (dest / "others").exists() is False

    # Testing DataChain File Contents
    edatachain_data = []
    for dqf in edatachain_files:
        assert dqf.is_file()
        edatachain_contents = yaml.safe_load(dqf.read_text())
        assert len(edatachain_contents) == 1
        edatachain_data.extend(edatachain_contents)

    assert len(edatachain_data) == 2
    data_cats1 = edatachain_data[0]
    data_dogs1 = edatachain_data[1]
    assert data_cats1["data-source"]["uri"] == sources[0].rstrip("/")
    assert data_dogs1["data-source"]["uri"] == sources[1].rstrip("/")
    assert len(data_cats1["files"]) == 2
    assert len(data_dogs1["files"]) == 4
    cat_files_by_name1 = {f["name"]: f for f in data_cats1["files"]}
    dog_files_by_name1 = {f["name"]: f for f in data_dogs1["files"]}

    # Directories should never be saved
    assert "others" not in dog_files_by_name1
    assert "dogs/others" not in dog_files_by_name1

    assert cat_files_by_name1["cat1"]["size"] == 4
    assert cat_files_by_name1["cat2"]["size"] == 4
    assert dog_files_by_name1["dog1"]["size"] == 4
    assert dog_files_by_name1["dog2"]["size"] == 3
    assert dog_files_by_name1["dog3"]["size"] == 4
    assert dog_files_by_name1["others/dog4"]["size"] == 4

    assert not dest.exists()

    with pytest.raises(FileNotFoundError):
        catalog.cp([str(dqf) for dqf in edatachain_files], str(dest), recursive=True)

    dest.mkdir()

    # Copy using these DataChain files as sources
    catalog.cp([str(dqf) for dqf in edatachain_files], str(dest), recursive=True)

    # Files should now be copied
    assert (dest / "cat1").read_text() == "meow"
    assert (dest / "cat2").read_text() == "mrow"
    assert (dest / "dog1").read_text() == "woof"
    assert (dest / "dog2").read_text() == "arf"
    assert (dest / "dog3").read_text() == "bark"
    assert (dest / "others" / "dog4").read_text() == "ruff"

    # Testing DataChain File Contents
    assert dest.with_suffix(".edatachain").is_file()
    edatachain_contents = yaml.safe_load(dest.with_suffix(".edatachain").read_text())
    assert len(edatachain_contents) == 2
    data_cats2 = edatachain_contents[0]
    data_dogs2 = edatachain_contents[1]
    assert data_cats2["data-source"]["uri"] == sources[0].rstrip("/")
    assert data_dogs2["data-source"]["uri"] == sources[1].rstrip("/")
    assert len(data_cats2["files"]) == 2
    assert len(data_dogs2["files"]) == 4
    cat_files_by_name2 = {f["name"]: f for f in data_cats2["files"]}
    dog_files_by_name2 = {f["name"]: f for f in data_dogs2["files"]}

    # Directories should never be saved
    assert "others" not in dog_files_by_name2
    assert "dogs/others" not in dog_files_by_name2

    assert cat_files_by_name2["cat1"]["size"] == 4
    assert cat_files_by_name2["cat2"]["size"] == 4
    assert dog_files_by_name2["dog1"]["size"] == 4
    assert dog_files_by_name2["dog2"]["size"] == 3
    assert dog_files_by_name2["dog3"]["size"] == 4
    assert dog_files_by_name2["others/dog4"]["size"] == 4


@pytest.mark.parametrize("cloud_type, version_aware", [("file", False)], indirect=True)
def test_cp_symlinks(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    catalog.client_config["use_symlinks"] = True
    src_uri = cloud_test_catalog.src_uri
    work_dir = cloud_test_catalog.working_dir
    dest = work_dir / "data"
    dest.mkdir()
    catalog.cp([f"{src_uri}/dogs/"], str(dest), recursive=True)

    assert (dest / "dog1").is_symlink()
    assert os.path.realpath(dest / "dog1") == str(
        cloud_test_catalog.src / "dogs" / "dog1"
    )
    assert (dest / "dog1").read_text() == "woof"
    assert (dest / "others" / "dog4").is_symlink()
    assert os.path.realpath(dest / "others" / "dog4") == str(
        cloud_test_catalog.src / "dogs" / "others" / "dog4"
    )
    assert (dest / "others" / "dog4").read_text() == "ruff"


def test_du(cloud_test_catalog, cloud_type):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    src_uri_path = src_uri
    if cloud_type == "file":
        src_uri_path = LocalFileSystem._strip_protocol(src_uri)
    expected_results = [
        (f"{src_uri_path}/cats/", 8),
        (f"{src_uri_path}/dogs/others/", 4),
        (f"{src_uri_path}/dogs/", 15),
        (f"{src_uri_path}/", 36),
    ]

    results = catalog.du([src_uri])
    assert set(results) == set(expected_results[3:])

    results = catalog.du([src_uri], depth=1)
    assert set(results) == set(expected_results[:1] + expected_results[2:])

    results = catalog.du([src_uri], depth=5)
    assert set(results) == set(expected_results)


def test_ls_glob(cloud_test_catalog):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    assert sorted(
        (source.node.name, [r[0] for r in results])
        for source, results in catalog.ls([f"{src_uri}/dogs/dog*"], fields=["name"])
    ) == [("dog1", ["dog1"]), ("dog2", ["dog2"]), ("dog3", ["dog3"])]


def test_ls_prefix_not_found(cloud_test_catalog):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog
    with pytest.raises(FileNotFoundError):
        list(catalog.ls([f"{src_uri}/bogus/"], fields=["name"]))


def clear_storages(catalog):
    ds = catalog.metastore
    ds.db.execute(ds._storages.delete())


@pytest.mark.parametrize("tree", [TARRED_TREE], indirect=True)
@pytest.mark.xfail(reason="Missing support for datasets in ls")
def test_ls_subobjects(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    create_tar_dataset(catalog, ctc.src_uri, "tarred")

    def do_ls(target):
        ((_, results),) = list(catalog.ls([target], fields=["name"]))
        results = list(results)
        result_set = {x[0] for x in results}
        assert len(result_set) == len(results)
        return result_set

    ds = "ds://tarred"
    assert do_ls(ds) == {"animals.tar"}
    assert do_ls(f"{ds}/animals.tar") == {"animals.tar"}
    assert do_ls(f"{ds}/animals.tar/dogs") == {
        "dog1",
        "dog2",
        "dog3",
        "others",
    }
    assert do_ls(f"{ds}/animals.tar/") == {"description", "cats", "dogs"}
    assert do_ls(f"{ds}/*.tar/") == {"description", "cats", "dogs"}
    assert do_ls(f"{ds}/*.tar/desc*") == {"description"}


def test_index_error(cloud_test_catalog):
    protocol = cloud_test_catalog.src_uri.split("://", 1)[0]
    # XXX: different clients raise inconsistent exceptions
    with pytest.raises(Exception):  # noqa: B017
        cloud_test_catalog.catalog.index([f"{protocol}://does_not_exist"])


def test_query(cloud_test_catalog, mock_popen_dataset_created):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri

    query_script = f"""\
    from datachain.query import C, DatasetQuery
    DatasetQuery({src_uri!r})
    """
    query_script = dedent(query_script)

    result = catalog.query(query_script, save=True)
    assert result.dataset
    assert_row_names(
        catalog,
        result.dataset,
        result.version,
        {
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )
    assert result.dataset.query_script == query_script
    assert result.dataset.sources == ""


def test_query_save_size(cloud_test_catalog, mock_popen_dataset_created):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri

    query_script = f"""\
    from datachain.query import C, DatasetQuery
    DatasetQuery({src_uri!r})
    """
    query_script = dedent(query_script)

    result = catalog.query(query_script, save=True)
    dataset_version = result.dataset.get_version(result.version)
    assert dataset_version.num_objects == 4
    assert dataset_version.size == 15


def test_query_fail_to_compile(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog

    query_script = "syntax error"

    with pytest.raises(QueryScriptCompileError):
        catalog.query(query_script)


def test_query_fail_wrong_dataset_name(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog

    query_script = """\
    from datachain.query import DatasetQuery
    DatasetQuery("s3://bucket-name")
    """
    query_script = dedent(query_script)

    with pytest.raises(
        ValueError, match="Cannot use ds_query_ prefix for dataset name"
    ):
        catalog.query(query_script, save_as="ds_query_dataset")


def test_query_subprocess_wrong_return_code(mock_popen, cloud_test_catalog):
    mock_popen.configure_mock(returncode=1)
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri

    query_script = f"""
from datachain.query import DatasetQuery, C
DatasetQuery('{src_uri}')
    """

    with pytest.raises(QueryScriptRunError) as exc_info:
        catalog.query(query_script)
        assert str(exc_info.value).startswith("Query script exited with error code 1")


def test_query_last_statement_not_expression(mock_popen, cloud_test_catalog):
    mock_popen.configure_mock(returncode=10)
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri

    query_script = f"""
from datachain.query import DatasetQuery, C
ds = DatasetQuery('{src_uri}')
    """

    with pytest.raises(QueryScriptCompileError) as exc_info:
        catalog.query(query_script)
        assert str(exc_info.value).startswith(
            "Query script failed to compile, "
            "reason: Last line in a script was not an expression"
        )


def test_query_last_statement_not_ds_query_instance(mock_popen, cloud_test_catalog):
    mock_popen.configure_mock(returncode=10)
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri

    query_script = f"""
from datachain.query import DatasetQuery, C
ds = DatasetQuery('{src_uri}')
5
    """

    with pytest.raises(QueryScriptRunError) as exc_info:
        catalog.query(query_script)
        assert str(exc_info.value).startswith(
            "Last line in a script was not an instance of DatasetQuery"
        )


def test_query_dataset_not_returned(mock_popen, cloud_test_catalog):
    mock_popen.configure_mock(stdout=io.StringIO("random str"))
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri

    query_script = f"""
from datachain.query import DatasetQuery, C
DatasetQuery('{src_uri}')
    """

    with pytest.raises(QueryScriptDatasetNotFound) as e:
        catalog.query(query_script, save=True)
    assert e.value.output == "random str"


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_storage_stats(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri

    with pytest.raises(StorageNotFoundError):
        catalog.storage_stats(src_uri)

    catalog.enlist_source(src_uri, ttl=1234)
    stats = catalog.storage_stats(src_uri)
    assert stats.num_objects == 7
    assert stats.size == 36

    catalog.enlist_source(f"{src_uri}/dogs/", ttl=1234, force_update=True)
    stats = catalog.storage_stats(src_uri)
    assert stats.num_objects == 4
    assert stats.size == 15

    catalog.enlist_source(f"{src_uri}/dogs/", ttl=1234)
    stats = catalog.storage_stats(src_uri)
    assert stats.num_objects == 4
    assert stats.size == 15


@pytest.mark.parametrize("from_cli", [False, True])
def test_garbage_collect(cloud_test_catalog, from_cli, capsys):
    catalog = cloud_test_catalog.catalog
    assert catalog.get_temp_table_names() == []
    temp_tables = ["tmp_vc12F", "udf_jh653", "ds_shadow_12345", "old_ds_shadow"]
    for t in temp_tables:
        catalog.warehouse.create_udf_table(t)
    assert set(catalog.get_temp_table_names()) == set(temp_tables)
    if from_cli:
        garbage_collect(catalog)
        captured = capsys.readouterr()
        assert captured.out == "Garbage collecting 4 tables.\n"
    else:
        catalog.cleanup_temp_tables(temp_tables)
    assert catalog.get_temp_table_names() == []


def test_get_file_signals(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    catalog.metastore.update_dataset_version(
        dogs_dataset,
        1,
        feature_schema={
            "name": "str",
            "age": "str",
            "f1": "File@1",
            "f2": "File@1",
        },
    )
    row = {
        "name": "Jon",
        "age": 25,
        "f1__source": "s3://first_bucket",
        "f1__name": "image1.jpg",
        "f2__source": "s3://second_bucket",
        "f2__name": "image2.jpg",
    }

    assert catalog.get_file_signals(dogs_dataset.name, 1, row) == {
        "source": "s3://first_bucket",
        "name": "image1.jpg",
    }


def test_get_file_signals_no_signals(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    catalog.metastore.update_dataset_version(
        dogs_dataset,
        1,
        feature_schema={
            "name": "str",
            "age": "str",
        },
    )
    row = {
        "name": "Jon",
        "age": 25,
    }

    assert catalog.get_file_signals(dogs_dataset.name, 1, row) is None


def test_open_object_no_file_signals(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    catalog.metastore.update_dataset_version(
        dogs_dataset,
        1,
        feature_schema={
            "name": "str",
            "age": "str",
        },
    )
    row = {
        "name": "Jon",
        "age": 25,
    }

    with pytest.raises(RuntimeError):
        assert catalog.open_object(dogs_dataset.name, 1, row)
