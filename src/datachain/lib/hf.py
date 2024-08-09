try:
    import datasets
    from datasets.features.features import string_to_arrow
    from datasets.features.image import image_to_bytes

except ImportError as exc:
    raise ImportError(
        "Missing dependencies for huggingface datasets:\n"
        "To install run:\n\n"
        "  pip install 'datasets'\n"
    ) from exc

from io import BytesIO
from typing import TYPE_CHECKING, Any

import PIL

from datachain.lib.arrow import arrow_type_mapper
from datachain.lib.data_model import DataModel, DataType, dict_to_data_model
from datachain.lib.udf import Generator

if TYPE_CHECKING:
    from pydantic import BaseModel


class HFClassLabel(DataModel):
    string: str
    integer: int


class HFImage(DataModel):
    img: bytes

    def read(self):
        return PIL.Image.open(BytesIO(self.img))


class HFAudio(DataModel):
    path: str
    array: list[float]
    sampling_rate: int


class HFGenerator(Generator):
    def __init__(self, ds: datasets.Dataset, output_schema: type["BaseModel"]):
        super().__init__()
        self.ds = ds
        self.output_schema = output_schema

    def process(self):
        if isinstance(self.ds, datasets.IterableDatasetDict):
            for split in self.ds:
                yield from self._process_ds(self.ds[split], split)
        else:
            yield from self._process_ds(self.ds)

    def _process_ds(self, ds: datasets.Dataset, split=False):
        for row in ds:
            output_dict = {}
            if split:
                output_dict["split"] = split
            for name, feat in ds.features.items():
                anno = self.output_schema.model_fields[name].annotation
                output_dict[name] = _convert_feature(row[name], feat, anno)
            yield self.output_schema(**output_dict)


def stream_dataset(path: str, **kwargs):
    return datasets.load_dataset(path, streaming=True, **kwargs)


def _convert_feature(val: Any, feat: Any, anno: Any) -> Any:
    if isinstance(feat, datasets.Value):
        return val
    if isinstance(feat, datasets.ClassLabel):
        return HFClassLabel(string=feat.names[val], integer=val)
    if isinstance(feat, datasets.Sequence):
        sdict = {}
        for sname in val:
            sfeat = feat.feature[sname]
            sanno = anno.model_fields[sname].annotation
            sdict[sname] = [_convert_feature(v, sfeat, sanno) for v in val]
        return anno(**sdict)
    if isinstance(
        feat, (datasets.Array2D, datasets.Array3D, datasets.Array4D, datasets.Array5D)
    ):
        return val.tolist()
    if isinstance(feat, datasets.Image):
        return HFImage(img=image_to_bytes(val))
    if isinstance(feat, datasets.Audio):
        return HFAudio(**val)


def get_output_schema(
    ds: datasets.Dataset, model_name: str = ""
) -> dict[str, DataType]:
    fields_dict = {}
    if isinstance(ds, datasets.IterableDatasetDict):
        fields_dict["split"] = str
        ds = ds[next(iter(ds.keys()))]
    for name, val in ds.features.items():
        fields_dict[name] = _feature_to_chain_type(name, val)  # type: ignore[assignment]
    return fields_dict  # type: ignore[return-value]


def _feature_to_chain_type(name: str, val: Any) -> type:  # noqa: PLR0911
    if isinstance(val, datasets.Value):
        return arrow_type_mapper(val.pa_type)
    if isinstance(val, datasets.ClassLabel):
        return HFClassLabel
    if isinstance(val, datasets.Sequence):
        sequence_dict = {}
        for sname, sval in val.feature.items():
            sequence_dict[sname] = list[_feature_to_chain_type(sname, sval)]  # type: ignore[misc]
        return dict_to_data_model(name, sequence_dict)
    if isinstance(val, datasets.Array2D):
        dtype = _feature_to_chain_type(name, string_to_arrow(val.dtype))  # type: ignore[misc]
        return list[list[dtype]]  # type: ignore[valid-type]
    if isinstance(val, datasets.Array3D):
        dtype = _feature_to_chain_type(name, string_to_arrow(val.dtype))  # type: ignore[misc]
        return list[list[list[dtype]]]  # type: ignore[valid-type]
    if isinstance(val, datasets.Array4D):
        dtype = _feature_to_chain_type(name, string_to_arrow(val.dtype))  # type: ignore[misc]
        return list[list[list[list[dtype]]]]  # type: ignore[valid-type]
    if isinstance(val, datasets.Array5D):
        dtype = _feature_to_chain_type(name, string_to_arrow(val.dtype))  # type: ignore[misc]
        return list[list[list[list[list[dtype]]]]]  # type: ignore[valid-type]
    if isinstance(val, datasets.Image):
        return HFImage
    if isinstance(val, datasets.Audio):
        return HFAudio
    raise TypeError(f"Unknown huggingface datasets type {type(val)}")
