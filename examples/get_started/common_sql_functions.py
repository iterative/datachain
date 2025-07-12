"""
Example demonstrating showing functions (manipulating strings, paths, arrays)
that are translated directly to SQL (vectorized). They don't require heavy compute,
fetching object into cluster, etc.
"""

import datachain as dc
from datachain import C
from datachain.func import array, greatest, least, path, string

chain = dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)

(
    chain.mutate(
        length=string.length(path.name(C("file.path"))),
        parts=string.split(path.name(C("file.path")), "."),
    )
    .select("file.path", "length", "parts")
    .show(5)
)

(
    chain.mutate(
        stem=path.file_stem(C("file.path")),
        ext=path.file_ext(C("file.path")),
    )
    .select("file.path", "stem", "ext")
    .show(5)
)

parts = string.split(path.name(C("file.path")), ".")
chain = chain.mutate(
    isdog=array.contains(parts, "dog"),
    iscat=array.contains(parts, "cat"),
)
chain.select("file.path", "isdog", "iscat").show(5)

chain = chain.mutate(
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
