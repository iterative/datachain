import logging

import datachain as dc
from datachain import Session

logging.basicConfig(level=logging.INFO)

# Dataset with variable session. Will persist.
session = Session("asVariable")
dv = dc.from_values(key=["a", "b", "c"], session=session)
dv.save("passed_as_argument")

# A datachain created in global context.
# This will be reverted back due to the global exception.
dc.from_values(key=["a", "b", "c"]).save("global_test_datachain_v1")

with Session("local"):
    # A datachain created in local context.
    # This will persist since error occur in global context.
    dc.from_values(key=["a", "b", "c"]).save("local_test_datachain")

try:
    with Session("local_failure"):
        # A datachain created in local context.
        # This will not persist since error occur in local context.
        dc.from_values(key=["a", "b", "c"]).save("local_test_datachain_v2")
        raise ValueError("Local failure class")
except ValueError:
    pass

# We return to global context. So, this will also be reverted.
dc.from_values(key=["a", "b", "c"]).save("global_error_class_v2")

raise Exception("This is a test exception")
