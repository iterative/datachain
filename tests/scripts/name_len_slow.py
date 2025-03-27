import sys
from time import sleep

import datachain as dc

if sys.platform == "win32":
    # This is needed for this process to accept a Ctrl-C event in Windows,
    # when run under pytest as a subprocess.
    # This is not needed when running normally from the command line.
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    if not kernel32.SetConsoleCtrlHandler(None, False):
        print("SetConsoleCtrlHandler error: ", ctypes.get_last_error(), file=sys.stderr)


def name_len(file):
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
    if file.path.endswith(".json"):
        return (-1,)
    return (len(file.path),)


# Save as a new dataset.
dc.read_storage(
    "gs://dvcx-datalakes/dogs-and-cats/",
    anon=True,
).filter(dc.C("file.path").glob("*cat*")).limit(3).settings(parallel=1).map(
    name_len, params=["file"], output={"name_len": int}
).save("name_len")
