import pandas as pd

import datachain.error
from datachain.lib.dc import DataChain
from datachain.lib.webdataset import WebDataset
from datachain.query.schema import C
from datachain.sql import literal
from datachain.sql.functions import array, greatest, least, string

name = "wds"
wds = DataChain(name=name)
try:
    df = wds.limit(3).to_pandas()
except datachain.error.DatasetNotFoundError:
    (
        DataChain("gs://dvcx-datacomp-small/shards", anon=True)
        .filter(C.name.glob("00000000.tar"))
        .generate(WebDataset())
        .save(name)
    )
    df = wds.limit(3).to_pandas()

print(df.columns.tolist())
columns = [
    "parent",
    "name",
    "vtype",
    "dir_type",
    "size",
    "caption",
    "url",
    "width",
    "height",
    "original_width",
    "original_height",
]
with pd.option_context("display.max_columns", None):
    print(df[columns])

filtered = (
    wds.filter(string.length(C.caption) > 5)
    .filter(array.length(string.split(C.caption, literal(" "))) > 2)
    .filter(least(C.original_width, C.original_height) > 200)
    .filter(
        greatest(C.original_width, C.original_height)
        / least(C.original_width, C.original_height)
        < 3.0
    )
)
filtered_df = filtered.limit(3).to_pandas()[columns]
with pd.option_context("display.max_columns", None):
    print(filtered_df)

print(f"wds count:      {wds.count():>6}")
print(f"filtered count: {filtered.count():>6}")
