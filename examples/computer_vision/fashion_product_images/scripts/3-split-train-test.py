from datachain.lib.dc import C, DataChain

# Define train_test_split function

print("\n# Define train_test_split function:")


def train_test_split(name) -> str:
    import random

    labels = ["train", "test", "val"]
    return random.choices(labels, weights=[0.7, 0.2, 0.1])[0]  # noqa: S311


# Add a signal (split)

print("\n# Add a signal (split):")
ds = (
    DataChain.from_dataset("fashion-product-images")
    .filter((C.masterCategory == "Apparel") & (C.subCategory == "Topwear"))
    .map(split=train_test_split, params=["name"], output=str)
    .save()
)

# Print splitting details

print("\n# Print splitting details:")
df = ds.to_pandas()
print(df.head(5))
print(df["split"].value_counts())


# Save train, test and val datasets

print("\n# Save train, test and val datasets:")
ds_train = ds.filter(C.split == "train").save("fashion-train")
ds_test = ds.filter(C.split == "test").save("fashion-test")
ds_val = ds.filter(C.split == "val").save("fashion-val")

# Print splitting details

print("Train dataset size: ", ds_train.to_pandas().shape)
print("Test dataset size: ", ds_test.to_pandas().shape)
print("Val dataset size: ", ds_val.to_pandas().shape)

ds_train.limit(3)
