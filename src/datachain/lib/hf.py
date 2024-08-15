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
    from datasets.features.features import string_to_arrow
    from datasets.features.image import image_to_bytes

except ImportError as exc:
    raise ImportError(
        "Missing dependencies for huggingface datasets:\n"
        "To install run:\n\n"
        "  pip install 'datasets'\n"
    ) from exc

from io import BytesIO
from typing import TYPE_CHECKING, Any, Union

import PIL

from datachain.lib.arrow import arrow_type_mapper
from datachain.lib.data_model import DataModel, DataType, dict_to_data_model
from datachain.lib.udf import Generator

if TYPE_CHECKING:
    from pydantic import BaseModel


HFDatasetType = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]


class HFClassLabel(DataModel):
    string: str
    integer: int

    def read(self):
        return self.integer


class HFImage(DataModel):
    path: str
    img: bytes

    def read(self):
        return PIL.Image.open(BytesIO(self.img))


class HFAudio(DataModel):
    path: str
    array: list[float]
    sampling_rate: int


class HFGenerator(Generator):
    def __init__(self, ds: HFDatasetType, output_schema: type["BaseModel"]):
        super().__init__()
        self.ds = ds
        self.output_schema = output_schema

    def process(self):
        if isinstance(self.ds, (DatasetDict, IterableDatasetDict)):
            for split in self.ds:
                yield from self._process_ds(self.ds[split], split)
        else:
            yield from self._process_ds(self.ds)

    def _process_ds(self, ds: Union[Dataset, IterableDataset], split=False):
        for row in ds:
            output_dict = {}
            if split:
                output_dict["split"] = split
            for name, feat in ds.features.items():
                anno = self.output_schema.model_fields[name].annotation
                output_dict[name] = _convert_feature(row[name], feat, anno)
            yield self.output_schema(**output_dict)


def stream_dataset(path: str, **kwargs):
    return load_dataset(path, streaming=True, **kwargs)


def _convert_feature(val: Any, feat: Any, anno: Any) -> Any:
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
                sdict[sname] = [_convert_feature(v, sfeat, sanno) for v in val[sname]]
            return anno(**sdict)
        return val
    if isinstance(feat, Image):
        return HFImage(img=image_to_bytes(val), path=val.filename)
    if isinstance(feat, Audio):
        return HFAudio(**val)


def get_output_schema(ds: HFDatasetType, model_name: str = "") -> dict[str, DataType]:
    fields_dict = {}
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        fields_dict["split"] = str
        ds = ds[next(iter(ds.keys()))]
    for name, val in ds.features.items():
        fields_dict[name] = _feature_to_chain_type(name, val)  # type: ignore[assignment]
    return fields_dict  # type: ignore[return-value]


def _feature_to_chain_type(name: str, val: Any) -> type:  # noqa: PLR0911
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
        return list[_feature_to_chain_type(name, val.feature)]  # type: ignore[arg-type,misc]
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
