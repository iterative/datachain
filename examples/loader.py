"""
A simple data loader example.

This downloads and displays the first 5 images of the dataset.
"""

from contextlib import closing

from datachain.catalog import get_catalog
from datachain.error import DatasetNotFoundError
from datachain.lib.param import Image
from datachain.query import C, DatasetQuery

catalog = get_catalog()
try:
    ds = catalog.get_dataset("cats")
except DatasetNotFoundError:
    ds = (
        DatasetQuery(
            path="gs://dvcx-datalakes/dogs-and-cats/",
            catalog=catalog,
        )
        .filter(C.name.glob("*cat*.jpg"))
        .save("cats")
    )


images = ds.limit(5).extract(Image(), cache=False)
with closing(images):
    for (img,) in images:
        img.show()
