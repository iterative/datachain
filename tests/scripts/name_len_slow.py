import sys
from time import sleep

from datachain.query import C, DatasetQuery, udf
from datachain.sql.types import Int

if sys.platform == "win32":
    # This is needed for this process to accept a Ctrl-C event in Windows,
    # when run under pytest as a subprocess.
    # This is not needed when running normally from the command line.
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    if not kernel32.SetConsoleCtrlHandler(None, False):
        print("SetConsoleCtrlHandler error: ", ctypes.get_last_error(), file=sys.stderr)


# Define the UDF:
@udf(
    ("name",),  # Columns consumed by the UDF.
    {
        "name_len": Int
    },  # Signals being returned by the UDF, with the signal name and type.
)
def name_len(name):
    # This is to avoid a sleep statement in the tests, so that the end-to-end test
    # knows when UDF processing has started, since we are testing canceling
    # UDF processing.
    # This is done to emulate a user waiting for processing that is stuck,
    # and pressing Ctrl-C to cancel the query script and UDF.
    print("UDF Processing Started")
    # Avoid any buffering so that the end-to-end test can react immediately.
    sys.stdout.flush()
    # Process very slowly to emulate a stuck script.
    sleep(1)
    if name.endswith(".json"):
        return (-1,)
    return (len(name),)


# Save as a new dataset.
DatasetQuery(
    path="gs://dvcx-datalakes/dogs-and-cats/",
    anon=True,
).filter(C.name.glob("*cat*")).add_signals(name_len, parallel=1).save("name_len")  # type: ignore[attr-defined]
