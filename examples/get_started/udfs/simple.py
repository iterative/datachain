import datachain as dc


# Define the UDF:
# DataChain figures out input and output types automatically
# based on the function signature and the data provided.
def path_len(path: str) -> int:
    if path.endswith(".json"):
        return -1
    return len(path)


if __name__ == "__main__":
    # Process all the files in the storage bucket, using the UDF
    # `read_storage` reads files from the specified path
    # and returns a DataChain object that has `File` objects
    (
        dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)
        .map(path_len=path_len, params=["file.path"])
        .show()
    )
