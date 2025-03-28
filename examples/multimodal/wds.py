import os

import datachain as dc
from datachain.func import path
from datachain.lib.webdataset import process_webdataset
from datachain.lib.webdataset_laion import WDSLaion, process_laion_meta

IMAGE_TARS = os.getenv(
    "IMAGE_TARS", "gs://datachain-demo/datacomp-small/shards/000000[0-5]*.tar"
)
PARQUET_METADATA = os.getenv(
    "PARQUET_METADATA", "gs://datachain-demo/datacomp-small/metadata/0020f*.parquet"
)
NPZ_METADATA = os.getenv(
    "NPZ_METADATA", "gs://datachain-demo/datacomp-small/metadata/0020f*.npz"
)

wds_images = (
    dc.read_storage(IMAGE_TARS, type="image")
    .settings(cache=True)
    .gen(laion=process_webdataset(spec=WDSLaion), params="file")
)

wds_with_pq = (
    dc.read_parquet(PARQUET_METADATA)
    .settings(cache=True)
    .merge(wds_images, on="uid", right_on="laion.json.uid", inner=True)
)

wds_npz = dc.read_storage(NPZ_METADATA).settings(cache=True).gen(emd=process_laion_meta)


res = wds_npz.merge(
    wds_with_pq,
    on=[path.file_stem(wds_npz.c("emd.file.path")), "emd.index"],
    right_on=[path.file_stem(wds_with_pq.c("source.file.path")), "source.index"],
    inner=True,
).save("wds")

res.show(5)
