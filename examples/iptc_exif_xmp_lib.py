from datachain.lib.iptc_exif_xmp import GetMetadata
from datachain.query import C, DatasetQuery

source = "gs://dvcx-datalakes/open-images-v6/"

if __name__ == "__main__":
    results = (
        DatasetQuery(source)
        .filter(C.name.glob("*.jpg"))
        .limit(10000)
        .add_signals(GetMetadata, parallel=True)
        .select("source", "xmp", "exif", "iptc", "error")
        .results()
    )
    print(*results, sep="\n")
