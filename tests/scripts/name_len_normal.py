from datachain.query import C, DatasetQuery, udf
from datachain.sql.types import Int


# Define the UDF:
@udf(
    ("name",),  # Columns consumed by the UDF.
    {
        "name_len": Int
    },  # Signals being returned by the UDF, with the signal name and type.
)
def name_len(name):
    if name.endswith(".json"):
        return (-1,)
    return (len(name),)


# Save as a new dataset.
DatasetQuery(
    path="gs://dvcx-datalakes/dogs-and-cats/",
    anon=True,
).filter(C.name.glob("*cat*")).add_signals(name_len, parallel=-1).order_by(  # type: ignore[attr-defined]
    "name"
).limit(2).save("name_len")

# Output the contents of the new dataset.
print(DatasetQuery(name="name_len").select(C.name, C.name_len).results())
