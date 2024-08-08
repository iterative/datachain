import os

from datachain import C, DataChain
from datachain.lib.webdataset import process_webdataset
from datachain.lib.webdataset_laion import WDSLaion, process_laion_meta
from datachain.sql.functions import path

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
    DataChain.from_storage(IMAGE_TARS)
    .settings(cache=True)
    .gen(laion=process_webdataset(spec=WDSLaion), params="file")
)

wds_with_pq = (
    DataChain.from_parquet(PARQUET_METADATA)
    .settings(cache=True)
    .merge(wds_images, on="uid", right_on="laion.json.uid", inner=True)
    .mutate(stem=path.file_stem(C("source.file.path")))
)

res = (
    DataChain.from_storage(NPZ_METADATA)
    .settings(cache=True)
    .gen(emd=process_laion_meta)
    .mutate(stem=path.file_stem(C("emd.file.path")))
    .merge(
        wds_with_pq,
        on=["stem", "emd.index"],
        right_on=["stem", "source.index"],
        inner=True,
    )
    .save("wds")
)

res.show(5)
