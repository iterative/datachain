import datachain as dc
import datachain.error
from datachain import func
from datachain.lib.webdataset import process_webdataset
from datachain.lib.webdataset_laion import WDSLaion

name = "wds"
try:
    wds = dc.read_dataset(name=name)
except datachain.error.DatasetNotFoundError:
    wds = (
        dc.read_storage("gs://datachain-demo/datacomp-small/shards")
        .filter(dc.C("file.path").glob("*/00000000.tar"))
        .settings(cache=True)
        .gen(laion=process_webdataset(spec=WDSLaion), params="file")
        .save(name)
    )

wds.print_schema()

filtered = (
    wds.filter(func.string.length("laion.txt") > 5)
    .filter(func.array.length(func.string.split("laion.txt", " ")) > 2)
    .filter(func.least("laion.json.original_width", "laion.json.original_height") > 200)
    .filter(
        func.greatest("laion.json.original_width", "laion.json.original_height")
        / func.least("laion.json.original_width", "laion.json.original_height")
        < 3.0
    )
    .persist()
)

filtered.show(3)

print(f"wds count:      {wds.count():>6}")
print(f"filtered count: {filtered.count():>6}")
