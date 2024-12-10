import re
from abc import ABC, abstractmethod
from collections.abc import Sequence


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


class DataChainColumnError(DataChainParamsError):
    def __init__(self, col_name: str, msg: str):
        super().__init__(f"Error for column {col_name}: {msg}")


def normalize_col_names(col_names: Sequence[str]) -> dict[str, str]:
    """Returns normalized_name -> original_name dict."""
    gen_col_counter = 0
    new_col_names = {}
    org_col_names = set(col_names)

    for org_column in col_names:
        new_column = org_column.lower()
        new_column = re.sub("[^0-9a-z]+", "_", new_column)
        new_column = new_column.strip("_")

        generated_column = new_column

        while (
            not generated_column.isidentifier()
            or generated_column in new_col_names
            or (generated_column != org_column and generated_column in org_col_names)
        ):
            if new_column:
                generated_column = f"c{gen_col_counter}_{new_column}"
            else:
                generated_column = f"c{gen_col_counter}"
            gen_col_counter += 1

        new_col_names[generated_column] = org_column

    return new_col_names
