"""
This is a simple UDF to demonstrate local parallel processing with multiprocessing.

In add_signals specify either parallel=-1 to use processes equal to the number
of CPUs/cores on your current machine, or parallel=N for N processes.
The default if parallel is not specified is to run single-threaded.

The UDF specified in add_signals will then be run in parallel across all these
worker processes, no other code changes are needed.

Benchmark results showed an almost 8X speedup on a MacBook Pro, using this test, with
parallel processing reducing execution time from a median of 377s to 48s total.

To install script dependencies: pip install tabulate
"""

from tabulate import tabulate

from datachain.query import C, DatasetQuery, udf
from datachain.sql.types import Int


# This is a simple single-threaded benchmark function to demonstrate the speedup
# that can be achieved with multiprocessing, by enabling parallel in add_signals.
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# Define the UDF:
@udf(
    ("name",),  # Columns consumed by the UDF.
    {
        "path_len": Int
    },  # Signals being returned by the UDF, with the signal name and type.
)
def name_len_benchmark(name):
    # Run the fibonacci benchmark as an example of a single-threaded CPU-bound UDF
    fibonacci(35)
    if name.endswith(".json"):
        return (-1,)
    return (len(name),)


# Save as a new dataset
DatasetQuery(
    path="gs://dvcx-datalakes/dogs-and-cats/",
    anon=True,
).filter(C.name.glob("*cat*")).add_signals(name_len_benchmark, parallel=-1).save(
    "cats_with_signal"
)

# Output the contents of the new dataset.
print(tabulate(DatasetQuery(name="cats_with_signal").results()[:10]))
