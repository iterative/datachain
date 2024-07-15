from datachain.lib.dc import C, DataChain
from datachain.lib.iptc_exif_xmp import image_description

source = "gs://dvcx-datalakes/open-images-v6/"

if __name__ == "__main__":
    results = (
        DataChain.from_storage(source, type="image")
        .filter(C("name").glob("*.jpg"))
        .limit(10000)
        .map(
            image_description,
            params=["file"],
            output={"xmp": dict, "exif": dict, "iptc": dict, "error": str},
        )
        .select("file.source", "xmp", "exif", "iptc", "error")
        .results()
    )
    print(*results, sep="\n")
