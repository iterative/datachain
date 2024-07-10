from collections.abc import Iterator
from typing import Optional

import numpy as np
from pydantic import Field

from datachain.lib.feature import Feature
from datachain.lib.file import File
from datachain.lib.webdataset import WDSBasic, WDSReadableSubclass


class Laion(WDSReadableSubclass):
    uid: str = Field(default="")
    face_bboxes: Optional[list[list[float]]] = Field(default=None)
    caption: Optional[str] = Field(default=None)
    url: Optional[str] = Field(default=None)
    key: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    original_width: Optional[int] = Field(default=None)
    original_height: Optional[int] = Field(default=None)
    exif: Optional[str] = Field(default=None)
    sha256: Optional[str] = Field(default=None)

    @staticmethod
    def _reader(builder, item):
        return Laion.model_validate_json(builder.read_text(item))


class WDSLaion(WDSBasic):
    txt: Optional[str] = Field(default=None)
    json: Laion  # type: ignore[assignment]


class LaionMeta(Feature):
    file: File
    index: Optional[int] = Field(default=None)
    b32_img: list[float] = Field(default=None)
    b32_txt: list[float] = Field(default=None)
    l14_img: list[float] = Field(default=None)
    l14_txt: list[float] = Field(default=None)
    dedup: list[float] = Field(default=None)


def process_laion_meta(file: File) -> Iterator[LaionMeta]:
    with file.open() as fd_npz:
        npz_file = np.load(fd_npz)
        b32_img = npz_file["b32_img"]
        b32_txt = npz_file["b32_txt"]
        l14_img = npz_file["l14_img"]
        l14_txt = npz_file["l14_txt"]
        dedup = npz_file["dedup"]

        for index in range(len(b32_img)):
            yield LaionMeta(
                file=file,
                index=index,
                b32_img=b32_img[index],
                b32_txt=b32_txt[index],
                l14_img=l14_img[index],
                l14_txt=l14_txt[index],
                dedup=dedup[index],
            )
