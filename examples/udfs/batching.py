# In some cases it is more expedient for a UDF to process several rows
# in one invocation. In such cases the UDFs need to be written in a
# slightly different way:
#  * They need to accept a single function parameter, which will be a
#    list of tuples of UDF inputs.
#  * They need to return a list of tuples of result values, one tuple
#    per row, in the same order as the input parameters.
# To install script dependencies: pip install tabulate
from sqlalchemy import Integer
from tabulate import tabulate

from datachain.query import C, DatasetQuery, udf


# Define the UDF:
@udf(
    ("parent", "name"),  # Columns consumed by the UDF.
    {
        "path_len": Integer
    },  # Signals being returned by the UDF, with the signal name and type.
    batch=10,
)
def name_len(names):
    return [(len(parent + name),) for (parent, name) in names]


if __name__ == "__main__":
    # Save as a new dataset
    DatasetQuery(path="gs://dvcx-datalakes/dogs-and-cats/").filter(
        C.name.glob("*cat*")
    ).add_signals(name_len).save("cats_with_signal")

    # Output the contents of the new dataset.
    print(tabulate(DatasetQuery(name="cats_with_signal").results()[:10]))
