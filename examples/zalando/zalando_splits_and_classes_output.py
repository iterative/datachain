import uuid

from zalando_splits_and_classes_ds import ds

from datachain.query import C, DatasetQuery

ds_name = uuid.uuid4().hex
ds.save(ds_name)
print(ds_name)
for row in (
    DatasetQuery(name=ds_name)
    .select(C.source, C.parent, C.name, C("class"), C("label"))
    .order_by(C.random)
    .limit(8)
    .results()
):
    print(row)
