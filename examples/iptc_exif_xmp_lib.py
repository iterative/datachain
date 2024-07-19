from datachain.lib.dc import C, DataChain
from datachain.lib.iptc_exif_xmp import image_description

source = "gs://dvcx-datalakes/open-images-v6/"

if __name__ == "__main__":
    (
        DataChain.from_storage(source, type="image")
        .settings(parallel=-1)
        .filter(C("name").glob("*.jpg"))
        .limit(10000)
        .map(
            image_description,
            params=["file"],
            output={"xmp": dict, "exif": dict, "iptc": dict, "error": str},
        )
        .select("file.name", "xmp", "exif", "iptc", "error")
        .filter((C("xmp") != "{}") | (C("exif") != "{}") | (C("iptc") != "{}"))
        .show()
    )
