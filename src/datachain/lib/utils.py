from abc import ABC, abstractmethod


class AbstractUDF(ABC):
    @abstractmethod
    def process(self, *args, **kwargs):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def teardown(self):
        pass


class DataChainError(Exception):
    def __init__(self, message):
        super().__init__(message)


class DataChainParamsError(DataChainError):
    def __init__(self, message):
        super().__init__(message)
