from datachain import DataChain


# Define the UDF:
def path_len(path):
    if path.endswith(".json"):
        return (-1,)
    return (len(path),)


if __name__ == "__main__":
    # Run in chain
    DataChain.from_storage(
        uri="gs://datachain-demo/dogs-and-cats/",
    ).map(
        path_len,
        params=["file.path"],
        output={"path_len": int},
    ).show()
