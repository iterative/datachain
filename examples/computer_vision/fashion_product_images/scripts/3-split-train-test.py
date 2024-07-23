from datachain import C, DataChain


def train_test_split(name) -> str:
    import random

    labels = ["train", "test", "val"]
    return random.choices(labels, weights=[0.7, 0.2, 0.1])[0]  # noqa: S311


print("\n# Add a signal (split):")
dc = (
    DataChain.from_dataset("fashion-product-images")
    .filter((C("masterCategory") == "Apparel") & (C("subCategory") == "Topwear"))
    .map(split=train_test_split, params=["file.name"], output=str)
    .save()
)

print("\n# Print splitting details:")
df = dc.to_pandas()
print(df.head(5))
print(df["split"].value_counts())


print("\n# Save train, test and val datasets:")
dc_train = dc.filter(C("split") == "train").save("fashion-train")
dc_test = dc.filter(C("split") == "test").save("fashion-test")
dc_val = dc.filter(C("split") == "val").save("fashion-val")

print("Train dataset size: ", dc_train.to_pandas().shape)
print("Test dataset size: ", dc_test.to_pandas().shape)
print("Val dataset size: ", dc_val.to_pandas().shape)

dc_train.limit(3)
