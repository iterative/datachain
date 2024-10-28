import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml
from fsspec.implementations.local import LocalFileSystem

from datachain import DataChain, File
from datachain.catalog import parse_edatachain_file
from datachain.cli import garbage_collect
from datachain.error import DatasetNotFoundError
from datachain.lib.listing import parse_listing_uri
from tests.data import ENTRIES
from tests.utils import DEFAULT_TREE, skip_if_not_sqlite, tree_from_path


def listing_stats(uri, catalog):
    list_dataset_name, _, _ = parse_listing_uri(
        uri, catalog.cache, catalog.client_config
    )
    dataset = catalog.get_dataset(list_dataset_name)
    return catalog.dataset_stats(dataset.name, dataset.latest_version)


@pytest.fixture
def pre_created_ds_name():
    return "pre_created_dataset"


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "gs", "azure"],
    indirect=True,
)
def test_find(cloud_test_catalog, cloud_type):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog
    dirs = ["cats/", "dogs/", "dogs/others/"]
    expected_paths = dirs + [entry.path for entry in ENTRIES]
    assert set(catalog.find([src_uri])) == {
        f"{src_uri}/{path}" for path in expected_paths
    }

    with pytest.raises(FileNotFoundError):
        set(catalog.find([f"{src_uri}/does_not_exist"]))


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "gs", "azure"],
    indirect=True,
)
def test_find_names_paths_size_type(cloud_test_catalog):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

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

    src_uri_path = src_uri
    if cloud_type == "file":
        src_uri_path = LocalFileSystem._strip_protocol(src_uri)

    assert set(
        catalog.find(
            [src_uri],
            names=["*cat*"],
            columns=["du", "name", "path", "size", "type"],
        )
    ) == {
        "\t".join(columns)
        for columns in [
            ["8", "cats", f"{src_uri_path}/cats/", "0", "d"],
            ["4", "cat1", f"{src_uri_path}/cats/cat1", "4", "f"],
            ["4", "cat2", f"{src_uri_path}/cats/cat2", "4", "f"],
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
    assert data["data-source"]["uri"] == src_uri.rstrip("/") + "/"
    expected_file_count = 7 if recursive else 1
    assert len(data["files"]) == expected_file_count
    files_by_name = {f["name"]: f for f in data["files"]}

    # Directories should never be saved
    assert "cats" not in files_by_name
    assert "dogs" not in files_by_name
    assert "others" not in files_by_name
    assert "dogs/others" not in files_by_name

    # Description is always copied (if anything is copied)
    prefix = "" if star or (recursive and not dir_exists) else "/"
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
@skip_if_not_sqlite
def test_cp_local_dataset(cloud_test_catalog, dogs_dataset):
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
    if not star and not slash and dir_exists:
        pytest.skip("Fix in https://github.com/iterative/datachain/issues/535")

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
    assert data["data-source"]["uri"] == src_uri.rstrip("/") + "/"
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
def test_cp_multi_subdir(cloud_test_catalog, recursive, star, slash, cloud_type):  # noqa: PLR0915
    if recursive and not star and not slash:
        pytest.skip("Fix in https://github.com/iterative/datachain/issues/535")

    if cloud_type == "file" and recursive and not star and slash:
        pytest.skip("Fix in https://github.com/iterative/datachain/issues/535")

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
    assert data_cats["data-source"]["uri"] == sources[0].rstrip("/") + "/"
    assert data_dogs["data-source"]["uri"] == sources[1].rstrip("/") + "/"
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
    src_uri = cloud_test_catalog.src_uri
    src_path = f"{src_uri}/dogs/others"
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog
    dest = working_dir / "data"

    catalog.cp([src_path], str(dest), recursive=True)

    # Testing DataChain File Contents
    assert dest.with_suffix(".edatachain").is_file()
    edatachain_contents = yaml.safe_load(dest.with_suffix(".edatachain").read_text())
    assert len(edatachain_contents) == 1
    data = edatachain_contents[0]
    assert data["data-source"]["uri"] == src_path.rstrip("/") + "/"
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

    (cloud_test_catalog.src / "foo").write_text("modified")
    dest = working_dir / "data2"
    dest.mkdir()
    catalog.cp([src_path], str(dest / "local"), no_edatachain_file=True)
    assert tree_from_path(dest) == {"local": "original"}

    # Since the old version cannot be found in storage or cache, it's an error.
    catalog.cache.clear()
    dest = working_dir / "data3"
    dest.mkdir()
    with pytest.raises(FileNotFoundError):
        catalog.cp([src_path], str(dest / "local"), no_edatachain_file=True)
    assert tree_from_path(dest) == {}

    catalog.index([src_path], update=True)
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
    assert data["data-source"]["uri"] == f"{cloud_test_catalog.src_uri}/dogs/"
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
    assert data["data-source"]["uri"] == f"{cloud_test_catalog.src_uri}/dogs/"
    assert edatachain_only_source["files"] == data["files"]


def test_cp_edatachain_file_sources(cloud_test_catalog):  # noqa: PLR0915
    pytest.skip("Fix in https://github.com/iterative/datachain/issues/535")
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
    assert data_cats1["data-source"]["uri"] == sources[0]
    assert data_dogs1["data-source"]["uri"] == sources[1].rstrip("*")
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
    assert data_cats2["data-source"]["uri"] == sources[0]
    assert data_dogs2["data-source"]["uri"] == sources[1].rstrip("*")
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


def test_index_error(cloud_test_catalog):
    protocol = cloud_test_catalog.src_uri.split("://", 1)[0]
    # XXX: different clients raise inconsistent exceptions
    with pytest.raises(Exception):  # noqa: B017
        cloud_test_catalog.catalog.index([f"{protocol}://does_not_exist"])


def test_dataset_stats(test_session):
    ids = [1, 2, 3]
    values = tuple(zip(["a", "b", "c"], [1, 2, 3]))

    ds1 = DataChain.from_values(
        ids=ids,
        file=[File(path=name, size=size) for name, size in values],
        session=test_session,
    ).save()
    dataset_version1 = test_session.catalog.get_dataset(ds1.name).get_version(1)
    assert dataset_version1.num_objects == 3
    assert dataset_version1.size == 6

    ds2 = DataChain.from_values(
        ids=ids,
        file1=[File(path=name, size=size) for name, size in values],
        file2=[File(path=name, size=size * 2) for name, size in values],
        session=test_session,
    ).save()
    dataset_version2 = test_session.catalog.get_dataset(ds2.name).get_version(1)
    assert dataset_version2.num_objects == 3
    assert dataset_version2.size == 18


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_listing_stats(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri

    with pytest.raises(DatasetNotFoundError):
        listing_stats(src_uri, catalog)

    catalog.enlist_source(src_uri)
    stats = listing_stats(src_uri, catalog)
    assert stats.num_objects == 7
    assert stats.size == 36

    catalog.enlist_source(f"{src_uri}/dogs/", update=True)
    stats = listing_stats(src_uri, catalog)
    assert stats.num_objects == 4
    assert stats.size == 15

    catalog.enlist_source(f"{src_uri}/dogs/")
    stats = listing_stats(src_uri, catalog)
    assert stats.num_objects == 4
    assert stats.size == 15


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_enlist_source_handles_slash(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    src_path = f"{src_uri}/dogs"

    catalog.enlist_source(src_path)
    stats = listing_stats(src_path, catalog)
    assert stats.num_objects == len(DEFAULT_TREE["dogs"])
    assert stats.size == 15

    src_path = f"{src_uri}/dogs"
    catalog.enlist_source(src_path, update=True)
    stats = listing_stats(src_path, catalog)
    assert stats.num_objects == len(DEFAULT_TREE["dogs"])
    assert stats.size == 15


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_enlist_source_handles_glob(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    src_path = f"{src_uri}/dogs/*.jpg"

    catalog.enlist_source(src_path)
    stats = listing_stats(src_path, catalog)

    assert stats.num_objects == len(DEFAULT_TREE["dogs"])
    assert stats.size == 15


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_enlist_source_handles_file(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    src_path = f"{src_uri}/dogs/dog1"

    catalog.enlist_source(src_path)
    stats = listing_stats(src_path, catalog)
    assert stats.num_objects == len(DEFAULT_TREE["dogs"])
    assert stats.size == 15


@pytest.mark.parametrize("from_cli", [False, True])
def test_garbage_collect(cloud_test_catalog, from_cli, capsys):
    catalog = cloud_test_catalog.catalog
    assert catalog.get_temp_table_names() == []
    temp_tables = [
        "tmp_vc12F",
        "udf_jh653",
    ]
    for t in temp_tables:
        catalog.warehouse.create_udf_table(name=t)
    assert set(catalog.get_temp_table_names()) == set(temp_tables)
    if from_cli:
        garbage_collect(catalog)
        captured = capsys.readouterr()
        assert captured.out == "Garbage collecting 2 tables.\n"
    else:
        catalog.cleanup_tables(temp_tables)
    assert catalog.get_temp_table_names() == []


def test_get_file_from_row(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    catalog.metastore.update_dataset_version(
        dogs_dataset,
        1,
        feature_schema={
            "name": "str",
            "age": "str",
            "f1": "File@v1",
            "f2": "File@v1",
        },
    )
    row = {
        "name": "Jon",
        "age": 25,
        "f1__source": "s3://first_bucket",
        "f1__path": "image1.jpg",
        "f2__source": "s3://second_bucket",
        "f2__path": "image2.jpg",
    }

    assert catalog.get_file_from_row(dogs_dataset.name, 1, row, "f1") == File(
        source="s3://first_bucket",
        path="image1.jpg",
    )
    assert catalog.get_file_from_row(dogs_dataset.name, 1, row, "f2") == File(
        source="s3://second_bucket",
        path="image2.jpg",
    )


def test_get_file_from_row_with_custom_types(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    catalog.metastore.update_dataset_version(
        dogs_dataset,
        1,
        feature_schema={
            "name": "str",
            "age": "str",
            "f1": "File@v1",
            "f2": "File@v1",
            "_custom_types": {
                "File@v1": {"source": "str", "path": "str"},
            },
        },
    )
    row = {
        "name": "Jon",
        "age": 25,
        "f1__source": "s3://first_bucket",
        "f1__path": "image1.jpg",
        "f2__source": "s3://second_bucket",
        "f2__path": "image2.jpg",
    }

    assert catalog.get_file_from_row(dogs_dataset.name, 1, row, "f1") == File(
        source="s3://first_bucket",
        path="image1.jpg",
    )


def test_get_file_from_row_no_signals(cloud_test_catalog, dogs_dataset):
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
        assert catalog.get_file_from_row(dogs_dataset.name, 1, row, "missing")
