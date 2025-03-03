# TextFile

`TextFile` is inherited from [`File`](file.md) with additional methods for working with text files.

`TextFile` is generated when a `DataChain` is created [from storage](../datachain.md#datachain.lib.dc.DataChain.from_storage), using `type="text"` param:

```python
from datachain import DataChain

dc = DataChain.from_storage("s3://bucket-name/", type="text")
```

::: datachain.lib.file.TextFile
