import datachain.error
from datachain import C, DataChain
from datachain.lib.webdataset import process_webdataset
from datachain.lib.webdataset_laion import WDSLaion
from datachain.sql import literal
from datachain.sql.functions import array, greatest, least, string

name = "wds"
try:
    wds = DataChain.from_dataset(name=name)
except datachain.error.DatasetNotFoundError:
    wds = (
        DataChain.from_storage("gs://datachain-demo/datacomp-small/shards")
        .filter(C("file.path").glob("*/00000000.tar"))
        .settings(cache=True)
        .gen(laion=process_webdataset(spec=WDSLaion), params="file")
        .save(name)
    )

wds.print_schema()

filtered = (
    wds.filter(string.length(C("laion.txt")) > 5)
    .filter(array.length(string.split(C("laion.txt"), literal(" "))) > 2)
    .filter(
        least(C("laion.json.original_width"), C("laion.json.original_height")) > 200
    )
    .filter(
        greatest(C("laion.json.original_width"), C("laion.json.original_height"))
        / least(C("laion.json.original_width"), C("laion.json.original_height"))
        < 3.0
    )
    .save()
)

filtered.show(3)

print(f"wds count:      {wds.count():>6}")
print(f"filtered count: {filtered.count():>6}")
