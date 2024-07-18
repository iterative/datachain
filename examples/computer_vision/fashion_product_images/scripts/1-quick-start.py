"""
# Getting Started with DataChain

Before you begin, ensure you have
- DataChain installed in your environment.
- Download the Fashion Product Images (Small) dataset (see README.md)
- Save data in the  `data` directory
    - `data/images`
    - `data/styles.csv`
"""

from datachain import C, DataChain

# Define the paths

DATA_PATH = "gs://datachain-demo/fashion-product-images"
ANNOTATIONS_PATH = "gs://datachain-demo/fashion-product-images/styles_clean.csv"

# Create a Dataset

print("\n# Create a Dataset")
ds = DataChain.from_storage(DATA_PATH, type="image").filter(C.name.glob("*.jpg"))
ds.show(3)

# Create a metadata DataChain

print("\n# Create a metadata DataChain")
ds_meta = DataChain.from_csv(ANNOTATIONS_PATH).select_except("source").save()
ds_meta.show(3)

# Merge the original image and metadata datachains

print("\n# Merge the original image and metadata datachains")
ds_annotated = ds.merge(ds_meta, on="name", right_on="filename")

# Save dataset

print("\n# Save dataset")
ds_annotated.save("fashion-product-images")


# Filtering Data

print("\n# Filtering Data")
ds = (
    DataChain.from_dataset(name="fashion-product-images")
    .filter(C.mastercategory == "Apparel")
    .filter(C.subcategory == "Topwear")
    .filter(C.season == "Summer")
)
print(ds.to_pandas().shape)


# NOTE: DataChain requires the  Last line to be an instance of DatasetQuery
ds.limit(3)
