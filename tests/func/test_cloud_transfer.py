from datachain import Session
from datachain.lib.dc import DataChain
from tests.conftest import get_cloud_test_catalog, make_cloud_server


def test_cross_cloud_transfer(
    tmp_upath_factory,
    tree,
    tmp_path,
    metastore,
    warehouse,
):
    # Setup cloud storage paths and servers
    azure_path = tmp_upath_factory.mktemp("azure", version_aware=False)
    azure_server = make_cloud_server(azure_path, "azure", tree)

    gcloud_path = tmp_upath_factory.mktemp("gs", version_aware=False)
    gcloud_server = make_cloud_server(gcloud_path, "gs", tree)

    # Initialize cloud catalogs
    azure_catalog = get_cloud_test_catalog(azure_server, tmp_path, metastore, warehouse)
    gcloud_catalog = get_cloud_test_catalog(
        gcloud_server, tmp_path, metastore, warehouse
    )

    # Define test file paths
    test_filename = "image_1.jpg"
    test_content = b"bytes"

    source_dir = f"{azure_catalog.src_uri}/source-test-images"
    source_file = f"{source_dir}/{test_filename}"

    dest_dir = f"{gcloud_catalog.src_uri}/destination-test-images"
    dest_file = f"{dest_dir}/{test_filename}"

    # Get cloud clients
    azure_client = azure_catalog.catalog.get_client(source_file)
    gcloud_client = gcloud_catalog.catalog.get_client(dest_file)

    try:
        # Create test file in Azure
        with azure_client.fs.open(source_file, "wb") as f:
            f.write(test_content)

        # Perform cross-cloud transfer
        combined_config = azure_server.client_config | gcloud_server.client_config
        with Session("testSession", client_config=combined_config, in_memory=True):
            datachain = DataChain.from_storage(source_dir)
            datachain.to_storage(dest_dir, placement="filename")

        # Verify transfer
        with gcloud_client.fs.open(dest_file, "rb") as f:
            assert f.read() == test_content

    finally:
        # Cleanup
        azure_client.fs.rm(source_dir, recursive=True)
        gcloud_client.fs.rm(dest_dir, recursive=True)
