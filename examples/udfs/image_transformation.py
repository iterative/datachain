import os

from PIL import ImageFile, ImageFilter
from tabulate import tabulate

from datachain.lib.image_transform import ImageTransform
from datachain.query import C, DatasetQuery

cloud_prefix = "s3://"  # for GCP just switch to "gs://"
bucket = "dvcx-50k-laion-files-writable"  # which bucket to use for both read and write
bucket_region = "us-east-2"  # no need to specify for GCP

file_type = "*.jpg"  # which files to use
blur_radius = 3  # how much to blur
filter_mod = 512  # how much of a subset of the data to use, i.e., 1/512

# only needed for AWS (no effect if using GCP)
os.environ["AWS_REGION"] = bucket_region
os.environ["AWS_DEFAULT_REGION"] = bucket_region

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Custom filters can be implemented with the ImageFilter abstract class
# https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.Filter
image_filter = ImageFilter.GaussianBlur(radius=blur_radius)

image_filter_udf = ImageTransform(
    image_filter=image_filter,
    bucket_name=bucket,
    prefix=cloud_prefix,
    output_folder="blur",
    file_prefix="blur_",
)

if __name__ == "__main__":
    data = (
        DatasetQuery(os.path.join(cloud_prefix, bucket))
        .filter(C.name.glob(file_type))
        .filter(C.random % filter_mod == 0)
        .generate(image_filter_udf)
        .results()
    )

    # Output the contents of the new dataset.
    print(tabulate(data))
