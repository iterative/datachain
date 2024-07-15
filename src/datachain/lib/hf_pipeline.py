import json

from transformers import pipeline

from datachain.lib.udf import Mapper


class Helper(Mapper):
    def __init__(self, model, device, **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    def setup(self):
        self.helper = pipeline(model=self.model, device=self.device)

    def process(self, file):
        imgs = file.get_value()
        result = self.helper(
            imgs,
            **self.kwargs,
        )
        return (json.dumps(result), "")
