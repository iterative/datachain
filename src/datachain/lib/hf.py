try:
    from datasets import (
        Array2D,
        Array3D,
        Array4D,
        Array5D,
        Audio,
        ClassLabel,
        Dataset,
        DatasetDict,
        Image,
        IterableDataset,
        IterableDatasetDict,
        Sequence,
        Value,
        load_dataset,
    )
    from datasets.features.features import Features, string_to_arrow
    from datasets.features.image import image_to_bytes

except ImportError as exc:
    raise ImportError(
        "Missing dependencies for huggingface datasets.\n"
        "To install run:\n\n"
        "  pip install 'datachain[hf]'\n"
    ) from exc

from io import BytesIO
from typing import TYPE_CHECKING, Any, Union

import PIL
from tqdm.auto import tqdm

from datachain.lib.arrow import arrow_type_mapper
from datachain.lib.data_model import DataModel, DataType, dict_to_data_model
from datachain.lib.udf import Generator

if TYPE_CHECKING:
    import pyarrow as pa
    from pydantic import BaseModel


HFDatasetType = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]


class HFClassLabel(DataModel):
    string: str
    integer: int

    def read(self):
        return self.integer


class HFImage(DataModel):
    img: bytes

    def read(self):
        return PIL.Image.open(BytesIO(self.img))


class HFAudio(DataModel):
    path: str
    array: list[float]
    sampling_rate: int


class HFGenerator(Generator):
    def __init__(
        self,
        ds: Union[str, HFDatasetType],
        output_schema: type["BaseModel"],
        *args,
        **kwargs,
    ):
        """
        Generator for chain from huggingface datasets.

        Parameters:

        ds : Path or name of the dataset to read from Hugging Face Hub,
            or an instance of `datasets.Dataset`-like object.
        output_schema : Pydantic model for validation.
        """
        super().__init__()
        self.ds = ds
        self.output_schema = output_schema
        self.args = args
        self.kwargs = kwargs

    def setup(self):
        self.ds_dict = stream_splits(self.ds, *self.args, **self.kwargs)

    def process(self, split: str = ""):
        desc = "Parsed Hugging Face dataset"
        ds = self.ds_dict[split]
        if split:
            desc += f" split '{split}'"
        with tqdm(desc=desc, unit=" rows", leave=False) as pbar:
            for row in ds:
                output_dict = {}
                if split and "split" in self.output_schema.model_fields:
                    output_dict["split"] = split
                for name, feat in ds.features.items():
                    anno = self.output_schema.model_fields[name].annotation
                    output_dict[name] = convert_feature(row[name], feat, anno)
                yield self.output_schema(**output_dict)
                pbar.update(1)


def stream_splits(ds: Union[str, HFDatasetType], *args, **kwargs):
    if isinstance(ds, str):
        kwargs["streaming"] = True
        ds = load_dataset(ds, *args, **kwargs)
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        return ds
    return {"": ds}


def convert_feature(val: Any, feat: Any, anno: Any) -> Any:  # noqa: PLR0911
    if isinstance(feat, (Value, Array2D, Array3D, Array4D, Array5D)):
        return val
    if isinstance(feat, ClassLabel):
        return HFClassLabel(string=feat.names[val], integer=val)
    if isinstance(feat, Sequence):
        if isinstance(feat.feature, dict):
            sdict = {}
            for sname in val:
                sfeat = feat.feature[sname]
                sanno = anno.model_fields[sname].annotation
                sdict[sname] = [convert_feature(v, sfeat, sanno) for v in val[sname]]
            return anno(**sdict)
        return val
    if isinstance(feat, Image):
        if isinstance(val, dict):
            return HFImage(img=val["bytes"])
        return HFImage(img=image_to_bytes(val))
    if isinstance(feat, Audio):
        return HFAudio(**val)


def get_output_schema(features: Features) -> dict[str, DataType]:
    """Generate UDF output schema from huggingface datasets features."""
    fields_dict = {}
    for name, val in features.items():
        fields_dict[name] = _feature_to_chain_type(name, val)
    return fields_dict


def _feature_to_chain_type(name: str, val: Any) -> DataType:  # noqa: PLR0911
    if isinstance(val, Value):
        return arrow_type_mapper(val.pa_type)
    if isinstance(val, ClassLabel):
        return HFClassLabel
    if isinstance(val, Sequence):
        if isinstance(val.feature, dict):
            sequence_dict = {}
            for sname, sval in val.feature.items():
                dtype = _feature_to_chain_type(sname, sval)
                sequence_dict[sname] = list[dtype]  # type: ignore[valid-type]
            return dict_to_data_model(name, sequence_dict)  # type: ignore[arg-type]
        return list[_feature_to_chain_type(name, val.feature)]  # type: ignore[arg-type,misc,return-value]
    if isinstance(val, Array2D):
        dtype = arrow_type_mapper(string_to_arrow(val.dtype))
        return list[list[dtype]]  # type: ignore[valid-type]
    if isinstance(val, Array3D):
        dtype = arrow_type_mapper(string_to_arrow(val.dtype))
        return list[list[list[dtype]]]  # type: ignore[valid-type]
    if isinstance(val, Array4D):
        dtype = arrow_type_mapper(string_to_arrow(val.dtype))
        return list[list[list[list[dtype]]]]  # type: ignore[valid-type]
    if isinstance(val, Array5D):
        dtype = arrow_type_mapper(string_to_arrow(val.dtype))
        return list[list[list[list[list[dtype]]]]]  # type: ignore[valid-type]
    if isinstance(val, Image):
        return HFImage
    if isinstance(val, Audio):
        return HFAudio
    raise TypeError(f"Unknown huggingface datasets type {type(val)}")


def schema_from_arrow(schema: "pa.Schema"):
    return Features.from_arrow_schema(schema)
