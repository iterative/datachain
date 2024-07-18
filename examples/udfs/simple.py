from datachain.lib.dc import DataChain


# Define the UDF:
def name_len(name):
    if name.endswith(".json"):
        return (-1,)
    return (len(name),)


if __name__ == "__main__":
    # Run in chain
    DataChain.from_storage(
        path="gs://dvcx-datalakes/dogs-and-cats/",
    ).map(
        name_len,
        params=["file.name"],
        output={"name_len": int},
    ).show()
