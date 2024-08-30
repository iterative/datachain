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
    from datasets.table import MemoryMappedTable

except ImportError as exc:
    raise ImportError(
        "Missing dependencies for huggingface datasets:\n"
        "To install run:\n\n"
        "  pip install 'datachain[hf]'\n"
    ) from exc

import inspect
from io import BytesIO
from typing import TYPE_CHECKING, Any, Union

import PIL
from tqdm import tqdm

from datachain.client.local import FileClient
from datachain.lib.arrow import arrow_type_mapper
from datachain.lib.data_model import DataModel, DataType, dict_to_data_model
from datachain.lib.file import File
from datachain.lib.udf import Generator

if TYPE_CHECKING:
    from datasets.features.features import Features
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


class HFFile(DataModel):
    file: File
    col: str
    row: int

    def read(self):
        tbl = _file_to_pyarrow(self.file)
        return tbl[self.col][self.row]


class HFImageFile(HFFile):
    file: File
    col: str
    row: int

    def read(self):
        raw = super().read()["bytes"].as_py()
        return PIL.Image.open(BytesIO(raw))


class HFGeneratorStream(Generator):
    def __init__(
        self,
        ds: Union[str, HFDatasetType],
        output_schema: type["BaseModel"],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.ds = ds
        self.output_schema = output_schema
        self.args = args
        self.kwargs = kwargs

    def setup(self):
        self.ds_dict = get_splits(self.ds, *self.args, streaming=True, **self.kwargs)

    def process(self, split: str = ""):
        desc = "Parsed Hugging Face dataset"
        ds = self.ds_dict[split]
        if split:
            desc += f" split '{split}'"
        with tqdm(desc=desc, unit=" rows") as pbar:
            for row in ds:
                output_dict = {}
                if split:
                    output_dict["split"] = split
                for name, feat in ds.features.items():
                    anno = self.output_schema.model_fields[name].annotation
                    output_dict[name] = _convert_feature(row[name], feat, anno)
                yield self.output_schema(**output_dict)
                pbar.update(1)


class HFGeneratorCache(Generator):
    def __init__(
        self,
        input_schema: "Features",
        output_schema: type["BaseModel"],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.args = args
        self.kwargs = kwargs

    def process(self, file: File, split: str = ""):
        desc = f"Parsed Hugging Face dataset file {file.name}"
        tbl = _file_to_pyarrow(file)

        row = 0
        with tqdm(desc=desc, unit=" rows") as pbar:
            for record_batch in tbl.to_batches():
                for record in record_batch.to_pylist():
                    output_dict = {}
                    if split:
                        output_dict["split"] = split
                    for name, feat in self.input_schema.items():
                        anno = self.output_schema.model_fields[name].annotation
                        if inspect.isclass(anno) and issubclass(anno, HFFile):
                            output_dict[name] = anno(file=file, row=row, col=name)  # type: ignore[assignment]
                        else:
                            output_dict[name] = _convert_feature(
                                record[name], feat, anno
                            )
                    yield self.output_schema(**output_dict)
                    row += 1
                pbar.update(len(record_batch))


def get_splits(ds: Union[str, HFDatasetType], *args, **kwargs):
    if isinstance(ds, str):
        ds = load_dataset(ds, *args, **kwargs)
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        return ds
    return {"": ds}


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
        return HFImage(img=image_to_bytes(val))
    if isinstance(feat, Audio):
        return HFAudio(**val)


def get_output_schema(
    ds: Union[Dataset, IterableDataset], model_name: str = "", stream: bool = True
) -> dict[str, DataType]:
    fields_dict = {}
    for name, val in ds.features.items():
        fields_dict[name] = _feature_to_chain_type(name, val, stream)  # type: ignore[assignment]
    return fields_dict  # type: ignore[return-value]


def _feature_to_chain_type(name: str, val: Any, stream: bool = True) -> type:  # noqa: PLR0911
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
        return HFImage if stream else HFImageFile
    if isinstance(val, Audio):
        return HFAudio if stream else HFFile
    raise TypeError(f"Unknown huggingface datasets type {type(val)}")


def _file_to_pyarrow(file: File):
    path = FileClient("/", {}, None).get_full_path(file.path)
    return MemoryMappedTable.from_file(path)
