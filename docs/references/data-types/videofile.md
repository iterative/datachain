# VideoFile

`VideoFile` extends [`File`](file.md) and provides additional methods for working with video files.

`VideoFile` instances are created when a `DataChain` is initialized [from storage](../datachain.md#datachain.lib.dc.DataChain.from_storage) with the `type="video"` parameter:

```python
from datachain import DataChain

dc = DataChain.from_storage("s3://bucket-name/", type="video")
```

There are additional models for working with video files:

- `VideoFrame` - represents a single frame of a video file.
- `VideoFragment` - represents a fragment of a video file.

These are virtual models that do not create physical files.
Instead, they are used to represent the data in the `VideoFile` these models are referring to.
If you need to save the data, you can use the `save` method of these models,
allowing you to save data locally or upload it to a storage service.

::: datachain.lib.file.VideoFile

::: datachain.lib.file.VideoFrame

::: datachain.lib.file.VideoFragment

::: datachain.lib.file.Video
