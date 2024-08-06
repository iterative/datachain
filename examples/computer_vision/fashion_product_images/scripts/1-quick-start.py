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

print("\n# Create a Dataset")
dc = DataChain.from_storage(DATA_PATH, type="image", anon=True).filter(
    C("file.name").glob("*.jpg")
)
dc.show(3)

print("\n# Create a metadata DataChain")
dc_meta = DataChain.from_csv(ANNOTATIONS_PATH).select_except("source").save()
dc_meta.show(3)

print("\n# Merge the original image and metadata datachains")
dc_annotated = dc.merge(dc_meta, on="file.name", right_on="filename")

print("\n# Save dataset")
dc_annotated.save("fashion-product-images")


print("\n# Filtering Data")
dc = DataChain.from_dataset(name="fashion-product-images").filter(
    C("mastercategory") == "Apparel"
    and C("subcategory") == "Topwear"
    and C("season") == "Summer"
)
dc.show()

# NOTE: Studio requires the last line to be an instance of DataChain
dc.save("fashion-summer-topwear-apparel")
