from datachain import C, DataChain
from datachain.lib.webdataset import process_webdataset
from datachain.lib.webdataset_laion import WDSLaion, process_laion_meta
from datachain.sql.functions import path

wds_images = (
    DataChain.from_storage("gs://datachain-demo/datacomp-small/shards")
    .filter(C("file.name").glob("000000[0-5]*.tar"))  # from *00.tar to *59.tar
    .settings(cache=True)
    .gen(laion=process_webdataset(spec=WDSLaion), params="file")
)

meta_pq = (
    DataChain.from_parquet("gs://datachain-demo/datacomp-small/metadata/0020f*.parquet")
    .settings(cache=True)
    .mutate(stem=path.file_stem(C("source.file.name")))
)

meta_emd = (
    DataChain.from_storage("gs://datachain-demo/datacomp-small/metadata/0020f*.npz")
    .settings(cache=True)
    .gen(emd=process_laion_meta)
    .mutate(stem=path.file_stem(C("emd.file.name")))
)

meta = meta_emd.merge(
    meta_pq,
    on=["stem", "emd.index"],
    right_on=["stem", "source.index"],
)

res = wds_images.merge(meta, on="laion.json.uid", right_on="uid").save("wds")

res.show(5)
