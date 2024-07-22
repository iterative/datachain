from datachain import C, DataChain
from datachain.sql import literal
from datachain.sql.functions import array, greatest, least, path, string


def num_chars_udf(name):
    parts = name.split(".")
    if len(parts) > 1:
        return (list(parts[1]),)
    return ([],)


ds = DataChain.from_storage("gs://datachain-demo/dogs-and-cats/")
ds.map(num_chars_udf, params=["file.name"], output={"num_chars": list[str]}).select(
    "file.name", "num_chars"
).show(5)

(
    ds.mutate(
        length=string.length(C("file.name")),
        parts=string.split(C("file.name"), literal(".")),
    )
    .select("file.name", "length", "parts")
    .show(5)
)

(
    ds.mutate(stem=path.file_stem(C("file.name")), ext=path.file_ext(C("file.name")))
    .select("file.name", "stem", "ext")
    .show(5)
)

(
    ds.mutate(
        a=array.length(string.split(C("file.parent"), literal("/"))),
        b=array.length(string.split(C("file.name"), literal("0"))),
    )
    .mutate(
        greatest=greatest(C("a"), C("b")),
        least=least(C("a"), C("b")),
    )
    .select("a", "b", "greatest", "least")
    .show(10)
)


"""
Expected output:

          file  num_chars
          name
0    cat.1.jpg        [1]
1   cat.1.json        [1]
2   cat.10.jpg     [1, 0]
3  cat.10.json     [1, 0]
4  cat.100.jpg  [1, 0, 0]

[Limited by 5 rows]
Processed: 400 rows [00:00, 23735.52 rows/s]
          file length            parts
          name
0    cat.1.jpg      9    [cat, 1, jpg]
1   cat.1.json     10   [cat, 1, json]
2   cat.10.jpg     10   [cat, 10, jpg]
3  cat.10.json     11  [cat, 10, json]
4  cat.100.jpg     11  [cat, 100, jpg]

[Limited by 5 rows]
Processed: 400 rows [00:00, 25456.28 rows/s]
          file     stem   ext
          name
0    cat.1.jpg    cat.1   jpg
1   cat.1.json    cat.1  json
2   cat.10.jpg   cat.10   jpg
3  cat.10.json   cat.10  json
4  cat.100.jpg  cat.100   jpg

[Limited by 5 rows]
Processed: 400 rows [00:00, 23276.15 rows/s]
   a  b  greatest  least
0  1  1         1      1
1  1  1         1      1
2  1  2         2      1
3  1  2         2      1
4  1  3         3      1
5  1  3         3      1
6  1  4         4      1
7  1  4         4      1
8  1  3         3      1
9  1  3         3      1

[Limited by 10 rows]
"""
