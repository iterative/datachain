from datachain.query import C, DatasetQuery, udf
from datachain.sql import literal
from datachain.sql.functions import array, greatest, least, path, string
from datachain.sql.types import Array, String


@udf(
    params=(C.name,),
    output={"num_chars": Array(String)},
)
def num_chars_udf(name):
    parts = name.split(".")
    if len(parts) > 1:
        return (list(parts[1]),)
    return ([],)


def show(dataset):
    print(*dataset.results(), sep="\n", end="\n\n")


ds = DatasetQuery("gs://dvcx-datalakes/dogs-and-cats/", anon=True)
show(ds.limit(5).add_signals(num_chars_udf).select(C.name, C.num_chars))
show(
    ds.limit(5).select(
        C.name,
        string.length(C.name),
        string.split(C.name, literal(".")),
    )
)
show(
    ds.limit(5).select(
        C.name,
        path.file_stem(C.name),
        path.file_ext(C.name),
    )
)
show(
    ds.limit(10)
    .mutate(
        a=array.length(string.split(C.parent, literal("/"))),
        b=array.length(string.split(C.name, literal("0"))),
    )
    .select(C.a, C.b, greatest(C.a, C.b), least(C.a, C.b))
)


"""
Expected output:
('', [])
('cat.1.jpg', ['1'])
('cat.1.json', ['1'])
('cat.10.jpg', ['1', '0'])
('cat.10.json', ['1', '0'])

('', 0, [''])
('cat.1.jpg', 9, ['cat', '1', 'jpg'])
('cat.1.json', 10, ['cat', '1', 'json'])
('cat.10.jpg', 10, ['cat', '10', 'jpg'])
('cat.10.json', 11, ['cat', '10', 'json'])

('', '', '')
('cat.1.jpg', 'cat.1', 'jpg')
('cat.1.json', 'cat.1', 'json')
('cat.10.jpg', 'cat.10', 'jpg')
('cat.10.json', 'cat.10', 'json')

(3, 1, 3, 1)
(3, 1, 3, 1)
(3, 1, 3, 1)
(3, 2, 3, 2)
(3, 2, 3, 2)
(3, 3, 3, 3)
(3, 3, 3, 3)
(3, 4, 4, 3)
(3, 4, 4, 3)
(3, 3, 3, 3)
"""
