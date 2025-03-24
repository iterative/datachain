# ImageFile

`ImageFile` is inherited from [`File`](file.md) with additional methods for working with image files.

`ImageFile` is generated when a `DataChain` is created [from storage](../datachain.md#datachain.lib.dc.storage.from_storage), using `type="image"` param:

```python
import datachain as dc

chain = dc.from_storage("s3://bucket-name/", type="image")
```

::: datachain.lib.file.ImageFile

::: datachain.lib.file.Image
