"""
This is a simple UDF to demonstrate local parallel processing with multiprocessing.

In add_signals specify either parallel=-1 to use processes equal to the number
of CPUs/cores on your current machine, or parallel=N for N processes.
The default if parallel is not specified is to run single-threaded.

The UDF specified in map will then be run in parallel across all these
worker processes, no other code changes are needed.
"""

from datachain.lib.dc import DataChain


# This is a simple single-threaded benchmark function to demonstrate the speedup
# that can be achieved with multiprocessing, by enabling parallel in add_signals.
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# Define the UDF:
def name_len_benchmark(name):
    # Run the fibonacci benchmark as an example of a single-threaded CPU-bound UDF
    fibonacci(35)
    if name.endswith(".json"):
        return (-1,)
    return len(name)


# Run in chain
DataChain.from_storage(
    path="gs://dvcx-datalakes/dogs-and-cats/",
).settings(parallel=-1).map(
    name_len_benchmark,
    params=["file.name"],
    output={"path_len": int},
).show()
