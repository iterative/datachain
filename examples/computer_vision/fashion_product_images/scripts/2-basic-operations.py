from datachain import C, DataChain

print("\n# Connect to a dataset:")
dc = DataChain.from_dataset("fashion-product-images")  # from 1-quick-start.py

print("\n# Filtering & Sorting:")
(
    dc.select(
        "file.parent",
        "file.name",
        "usage",
        "season",
        "year",
        "gender",
        "mastercategory",
        "subcategory",
        "articletype",
        "basecolour",
        "productdisplayname",
    )
    .filter(C("usage") == "Casual" and C("season") == "Summer")
    .order_by("year")
    .group_by("gender")
    .show()
)


print("\n# Add signals (columns) with map() method:")
dc_len = DataChain.from_dataset("fashion-product-images").map(
    prod_name_length=lambda name: len(name),
    params=["file.name"],
    output=int,
)

dc_len.show(3)

print("\n# Save a dataset (version):")
dc_len.save("fashion-tmp")

print("\n# Save a new version (with prod_name_length_2 column):")
(
    DataChain(name="fashion-summer-topwear-apparel")
    .map(prod_name_length_2=lambda name: len(name), params=["file.name"], output=int)
    .save("fashion-tmp")
)

# Load the latest version and show the first 3 rows -
print("\n# Load the latest version and show the first 3 rows:")
DataChain(name="fashion-tmp").limit(3)
