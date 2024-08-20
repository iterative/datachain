from datachain import C, DataChain
from datachain.sql import literal
from datachain.sql.functions import array, greatest, least, path, string


def num_chars_udf(file):
    parts = file.name.split(".")
    if len(parts) > 1:
        return (list(parts[1]),)
    return ([],)


dc = DataChain.from_storage("gs://datachain-demo/dogs-and-cats/")
dc.map(num_chars_udf, params=["file"], output={"num_chars": list[str]}).select(
    "file.path", "num_chars"
).show(5)

(
    dc.mutate(
        length=string.length(path.name(C("file.path"))),
        parts=string.split(path.name(C("file.path")), literal(".")),
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


chain = dc.mutate(
    a=array.length(string.split(C("file.path"), literal("/"))),
    b=array.length(string.split(path.name(C("file.path")), literal("0"))),
)

(
    chain.mutate(
        greatest=greatest(chain.column("a"), C("b")),
        least=least(chain.column("a"), C("b")),
    )
    .select("a", "b", "greatest", "least")
    .show(10)
)


"""
Expected output:

                        file  num_chars
                        path
0    dogs-and-cats/cat.1.jpg        [1]
1   dogs-and-cats/cat.1.json        [1]
2   dogs-and-cats/cat.10.jpg     [1, 0]
3  dogs-and-cats/cat.10.json     [1, 0]
4  dogs-and-cats/cat.100.jpg  [1, 0, 0]

[Limited by 5 rows]
Processed: 400 rows [00:00, 14314.30 rows/s]
                        file length            parts
                        path
0    dogs-and-cats/cat.1.jpg      9    [cat, 1, jpg]
1   dogs-and-cats/cat.1.json     10   [cat, 1, json]
2   dogs-and-cats/cat.10.jpg     10   [cat, 10, jpg]
3  dogs-and-cats/cat.10.json     11  [cat, 10, json]
4  dogs-and-cats/cat.100.jpg     11  [cat, 100, jpg]

[Limited by 5 rows]
Processed: 400 rows [00:00, 16364.66 rows/s]
                        file     stem   ext
                        path
0    dogs-and-cats/cat.1.jpg    cat.1   jpg
1   dogs-and-cats/cat.1.json    cat.1  json
2   dogs-and-cats/cat.10.jpg   cat.10   jpg
3  dogs-and-cats/cat.10.json   cat.10  json
4  dogs-and-cats/cat.100.jpg  cat.100   jpg

[Limited by 5 rows]
Processed: 400 rows [00:00, 16496.93 rows/s]
   a  b  greatest  least
0  2  1         2      1
1  2  1         2      1
2  2  2         2      2
3  2  2         2      2
4  2  3         3      2
5  2  3         3      2
6  2  4         4      2
7  2  4         4      2
8  2  3         3      2
9  2  3         3      2

[Limited by 10 rows]

"""
