import io

from fsspec.callbacks import DEFAULT_CALLBACK, Callback


class FileWrapper(io.RawIOBase):
    """Instrumented wrapper around an existing file object.

    It wraps the file's read() method to update the callback with the number of
    bytes read.

    It assumes exclusive access to the underlying file object and closes it when it
    gets closed itself.
    """

    def __init__(self, fileobj, callback: Callback = DEFAULT_CALLBACK):
        self.fileobj = fileobj
        self.callback = callback

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return self.fileobj.seekable()

    def tell(self):
        """Return the current file position."""
        return self.fileobj.tell()

    def seek(self, position, whence=io.SEEK_SET):
        """Seek to a position in the file."""
        return self.fileobj.seek(position, whence)

    def readinto(self, b) -> int:
        res = self.fileobj.readinto(b)
        self.callback.relative_update(res)
        return res

    def close(self):
        self.fileobj.close()
        super().close()
