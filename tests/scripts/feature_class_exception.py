import logging

from datachain.lib.dc import DataChain

logging.basicConfig(level=logging.INFO)


ds_name = "feature_class_error"
ds = DataChain.from_values(key=["a", "b", "c"])
ds.save(ds_name)
raise Exception("This is a test exception")
