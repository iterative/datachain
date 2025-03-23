from datachain import C, DataChain
from datachain.func import array, greatest, least, path, string


def num_chars_udf(file):
    parts = file.name.split(".")
    if len(parts) > 1:
        return (list(parts[1]),)
    return ([],)


dc = DataChain.from_storage("gs://datachain-demo/dogs-and-cats/", anon=True)
dc.map(num_chars_udf, params=["file"], output={"num_chars": list[str]}).select(
    "file.path", "num_chars"
).show(5)

(
    dc.mutate(
        length=string.length(path.name(C("file.path"))),
        parts=string.split(path.name(C("file.path")), "."),
    )
    .select("file.path", "length", "parts")
    .show(5)
)

(
    dc.mutate(
        stem=path.file_stem(C("file.path")),
        ext=path.file_ext(C("file.path")),
    )
    .select("file.path", "stem", "ext")
    .show(5)
)

parts = string.split(path.name(C("file.path")), ".")
chain = dc.mutate(
    isdog=array.contains(parts, "dog"),
    iscat=array.contains(parts, "cat"),
)
chain.select("file.path", "isdog", "iscat").show(5)

chain = dc.mutate(
    a=array.length(string.split("file.path", "/")),
    b=array.length(string.split(path.name("file.path"), "0")),
)

(
    chain.mutate(
        greatest=greatest(chain.column("a"), C("b")),
        least=least(chain.column("a"), C("b")),
    )
    .select("a", "b", "greatest", "least")
    .show(10)
)
