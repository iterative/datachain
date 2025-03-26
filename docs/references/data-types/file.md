# File

`File` is a special [`DataModel`](index.md#datachain.lib.data_model.DataModel),
which is automatically generated when a `DataChain` is created from files,
such as in [`dc.from_storage`](../datachain.md#datachain.lib.dc.storage.from_storage):

```python
import datachain as dc

chain = dc.from_storage("gs://datachain-demo/dogs-and-cats")
chain.print_schema()
```

Output:

```
file: File@v1
    source: str
    path: str
    size: int
    version: str
    etag: str
    is_latest: bool
    last_modified: datetime
    location: Union[dict, list[dict], NoneType]
```

`File` classes include various metadata fields describing the underlying file,
along with methods to read and manipulate file contents.

::: datachain.lib.file.File

::: datachain.lib.file.FileError

::: datachain.lib.file.TarVFile
