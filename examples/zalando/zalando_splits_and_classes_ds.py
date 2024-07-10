from datachain.query import C, DatasetQuery
from datachain.sql.functions import path

source_path = "gs://dvcx-zalando-hd-resized/zalando-hd-resized/"
ds = (
    DatasetQuery(source_path)
    .filter(C.name.glob("*.jpg"))
    .mutate(**{"class": path.name(C.parent), "label": path.name(path.parent(C.parent))})
)
