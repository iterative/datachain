from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
)

from datachain.lib.data_model import dict_to_data_model
from datachain.query import Session

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from datachain.lib.data_model import DataType
    from datachain.lib.hf import HFDatasetType

    from .datachain import DataChain

    P = ParamSpec("P")


def from_hf(
    dataset: Union[str, "HFDatasetType"],
    *args,
    session: Optional[Session] = None,
    settings: Optional[dict] = None,
    object_name: str = "",
    model_name: str = "",
    **kwargs,
) -> "DataChain":
    """Generate chain from huggingface hub dataset.

    Parameters:
        dataset : Path or name of the dataset to read from Hugging Face Hub,
            or an instance of `datasets.Dataset`-like object.
        session : Session to use for the chain.
        settings : Settings to use for the chain.
        object_name : Generated object column name.
        model_name : Generated model name.
        kwargs : Parameters to pass to datasets.load_dataset.

    Example:
        Load from Hugging Face Hub:
        ```py
        import datachain as dc
        chain = dc.from_hf("beans", split="train")
        ```

        Generate chain from loaded dataset:
        ```py
        from datasets import load_dataset
        ds = load_dataset("beans", split="train")
        import datachain as dc
        chain = dc.from_hf(ds)
        ```
    """
    from datachain.lib.hf import HFGenerator, get_output_schema, stream_splits

    from .values import from_values

    output: dict[str, DataType] = {}
    ds_dict = stream_splits(dataset, *args, **kwargs)
    if len(ds_dict) > 1:
        output = {"split": str}

    model_name = model_name or object_name or ""
    hf_features = next(iter(ds_dict.values())).features
    output = output | get_output_schema(hf_features)
    model = dict_to_data_model(model_name, output)
    if object_name:
        output = {object_name: model}

    chain = from_values(split=list(ds_dict.keys()), session=session, settings=settings)
    return chain.gen(HFGenerator(dataset, model, *args, **kwargs), output=output)
