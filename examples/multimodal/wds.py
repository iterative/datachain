from datachain import C, DataChain
from datachain.lib.webdataset import process_webdataset
from datachain.lib.webdataset_laion import WDSLaion, process_laion_meta

wds = (
    DataChain.from_storage("gs://datachain-demo/datacomp-small/shards")
    .filter(C("file.name").glob("00000000.tar"))
    .settings(cache=True)
    .gen(laion=process_webdataset(spec=WDSLaion), params="file")
    .save()  # materialize chain to avoid downloading data multiple times
)

meta_pq = (
    DataChain.from_parquet("gs://datachain-demo/datacomp-small/metadata/0020f*.parquet")
    .filter(
        C("uid").in_(values[0] for values in wds.select("laion.json.uid").collect())
    )
    .map(stem=lambda file: file.get_file_stem(), params=["source.file"], output=str)
    .save()
)

meta_emd = (
    DataChain.from_storage("gs://datachain-demo/datacomp-small/metadata/0020f*.npz")
    .gen(emd=process_laion_meta)
    .filter(
        C("emd.index").in_(
            values[0] for values in meta_pq.select("source.index").collect()
        )
    )
    .map(stem=lambda file: file.get_file_stem(), params=["emd.file"], output=str)
)


meta = meta_emd.merge(
    meta_pq,
    on=["stem", "emd.index"],
    right_on=["stem", "source.index"],
)

res = wds.merge(meta, on="laion.json.uid", right_on="uid")

res.show(3)
