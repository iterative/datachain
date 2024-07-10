import os

import fsspec
from PIL import Image

from datachain.catalog import get_catalog
from datachain.query import DatasetRow, Object, udf


def load_image(raw):
    img = Image.open(raw)
    img.load()
    return img


@udf(
    output=DatasetRow.schema,
    params=(Object(load_image), *tuple(DatasetRow.schema.keys())),
)
class ImageTransform:
    def __init__(
        self,
        *,
        image_filter,
        bucket_name,
        prefix,
        output_folder,
        file_prefix="",
        vtype="",
    ):
        # Once we fix the UDF decorator situation, it would make more sense to put this
        # into a child class and make apply_filter an abstractmethod.
        self.image_filter = image_filter
        self.folder_name = output_folder
        self.file_prefix = file_prefix
        self.prefix = prefix
        self.vtype = vtype

        catalog = get_catalog()
        self.client, _ = catalog.parse_url(os.path.join(self.prefix, bucket_name))

    def apply_filter(self, image):
        return image.filter(self.image_filter)

    def save(self, image, source, name, format):
        # Make name for new image
        new_name = f"{self.file_prefix}{name}"

        # Do writeback
        blob_name = os.path.join(self.folder_name, new_name)
        urlpath = os.path.join(source, blob_name)
        cloud_file = fsspec.open(urlpath=urlpath, mode="wb")
        with cloud_file as fp:
            image.save(fp, format=format)

        # Get the blob info
        info_ = self.client.fs.info(urlpath)
        info = self.client.convert_info(info_, self.folder_name)
        info.name = new_name
        return info

    def __call__(
        self,
        image,
        *args,
    ):
        # Build a dict from row contents
        record = dict(zip(DatasetRow.schema.keys(), args))
        del record["random"]  # random will be populated automatically
        record["is_latest"] = record["is_latest"] > 0  # needs to be a bool

        # yield same row back
        yield DatasetRow.create(**record)

        # Don't apply the filter twice
        if record["parent"] == self.folder_name:
            return

        # Apply the filter
        image_b = self.apply_filter(image)

        # Save the image and get the cloud object info
        entry = self.save(
            image_b, record["source"], name=record["name"], format=image.format
        )

        # Build the new row
        yield DatasetRow.create(
            name=entry.name,
            source=record["source"],
            parent=self.folder_name,
            size=entry.size,
            location=record["name"]
            if not record["parent"]
            else f"{record['parent']}/{record['name']}",
            vtype=self.vtype,
            dir_type=record["dir_type"],
            owner_name=entry.owner_name,
            owner_id=entry.owner_id,
            is_latest=record["is_latest"],
            last_modified=entry.last_modified,
            version=entry.version,
            etag=entry.etag,
        )
