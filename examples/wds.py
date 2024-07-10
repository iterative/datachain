import pandas as pd

from datachain.lib.dc import C, DataChain
from datachain.lib.webdataset import process_webdataset
from datachain.lib.webdataset_laion import WDSLaion, process_laion_meta

wds = (
    DataChain.from_storage("gs://dvcx-datacomp-small/shards")
    .filter(C.name.glob("00000000.tar"))
    .settings(cache=True)
    .gen(laion=process_webdataset(spec=WDSLaion), params="file")
)

meta_emd = (
    DataChain.from_storage("gs://dvcx-datacomp-small/metadata")
    .filter(C.name.glob("0020f*.npz"))
    .gen(emd=process_laion_meta)
    .map(stem=lambda file: file.get_file_stem(), params=["emd.file"], output=str)
)

meta_pq = (
    DataChain.from_storage("gs://dvcx-datacomp-small/metadata")
    .filter(C.name.glob("0020f*.parquet"))
    .parse_parquet()
    .map(stem=lambda file: file.get_file_stem(), params=["source.file"], output=str)
)

meta = meta_emd.merge(
    meta_pq, on=["stem", "emd.index"], right_on=["stem", "source.index"]
)

res = wds.merge(meta, on="laion.json.uid", right_on="uid")

df = res.limit(10).to_pandas()
with pd.option_context("display.max_columns", None):
    print(df)
