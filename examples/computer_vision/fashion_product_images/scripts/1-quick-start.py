"""
# Getting Started with DataChain

Before you begin, ensure you have
- DataChain installed in your environment.
- Download the Fashion Product Images (Small) dataset (see README.md)
- Save data in the  `data` directory
    - `data/images`
    - `data/styles.csv`
"""

import pandas as pd

from datachain.lib.dc import C, DataChain

DATA_PATH = "data/images"
ANNOTATIONS_PATH = "data/styles.csv"


# Create a Dataset

print("\n# Create a Dataset:")
ds = DataChain.from_storage(DATA_PATH, type="image").filter(C.name.glob("*.jpg"))
print(ds.show(3))

# Preview as a Pandas DataFrame

print("\n# Preview as a Pandas DataFrame:")
df = ds.to_pandas()
print(df.shape)
print(df.head(3))


# Create a Metadata DataChain

print("\n# Add Metadata:")
annotations = pd.read_csv(
    ANNOTATIONS_PATH,
    usecols=[
        "id",
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "year",
        "usage",
        "productDisplayName",
    ],
)

# Preprocess columns

annotations["baseColour"] = annotations["baseColour"].fillna("")
annotations["season"] = annotations["season"].fillna("")
annotations["usage"] = annotations["usage"].fillna("")
annotations["productDisplayName"] = annotations["productDisplayName"].fillna("")
annotations["filename"] = annotations["id"].apply(lambda s: str(s) + ".jpg")
annotations = annotations.drop("id", axis=1)

# Create a metadata DataChain

ds_meta = DataChain.from_pandas(annotations)
ds_meta.show(3)

# Merge the original image and metadata datachains

print("\n# Merge the original image and metadata datachains:")
ds_annotated = ds.merge(ds_meta, on="name", right_on="filename")

# Save dataset

print("\n# Save dataset:")
ds_annotated.save("fashion-product-images")


# Filtering Data

print("\n# Filtering Data:")
ds = (
    DataChain.from_dataset(name="fashion-product-images")
    .filter(C.mastercategory == "Apparel")
    .filter(C.subcategory == "Topwear")
    .filter(C.season == "Summer")
)
print(ds.to_pandas().shape)


# NOTE: DataChain requires the  Last line to be an instance of DatasetQuery
ds.limit(3)
