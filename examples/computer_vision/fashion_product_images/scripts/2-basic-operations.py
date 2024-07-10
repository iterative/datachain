from datachain.lib.dc import C, DataChain

# Create a dataset

print("\n# Connect to a dataset:")
ds = DataChain.from_dataset("fashion-product-images")

# Filtering & Sorting

print("\n# Filtering & Sorting:")
(
    DataChain.from_dataset("fashion-product-images")
    .select("parent", "name", "usage", "season", "year", "gender")
    .filter(C.usage == "Casual" and C.season == "Summer")
    .order_by("year")
    .group_by("gender")
    .to_pandas()
)

# Add signals (columns) with map() method

print("\n# Add signals (columns) with map() method:")
(
    DataChain.from_dataset("fashion-product-images")
    .map(prod_name_length=lambda name: len(name), output=int)
    .show(3)
)


# Save a dataset (version)

print("\n# Save a dataset (version):")
(
    DataChain.from_dataset(name="fashion-topwear")
    .map(prod_name_length=lambda name: len(name), output=int)
    .save("fashion-tmp")
)

# Save a new version  (with "prod_name_length_2" column)

print("\n# Save a new version  (with prod_name_length_2 column):")
(
    DataChain(name="fashion-topwear")
    .map(prod_name_length_2=lambda name: len(name), output=int)
    .save("fashion-tmp")
)

# Load the latest version and show the first 3 rows

print("\n# Load the latest version and show the first 3 rows:")
DataChain(name="fashion-tmp").limit(3)
