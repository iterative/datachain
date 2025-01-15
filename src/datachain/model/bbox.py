from pydantic import Field

from datachain.lib.data_model import DataModel


class BBox(DataModel):
    """
    A data model for representing bounding box.

    Attributes:
        title (str): The title of the bounding box.
        coords (list[int]): The coordinates of the bounding box.

    The bounding box is defined by two points:
        - (x1, y1): The top-left corner of the box.
        - (x2, y2): The bottom-right corner of the box.
    """

    title: str = Field(default="")
    coords: list[int] = Field(default=[])

    @staticmethod
    def from_list(coords: list[float], title: str = "") -> "BBox":
        assert len(coords) == 4, "Bounding box must be a list of 4 coordinates."
        assert all(isinstance(value, (int, float)) for value in coords), (
            "Bounding box coordinates must be floats or integers."
        )
        return BBox(
            title=title,
            coords=[round(c) for c in coords],
        )

    @staticmethod
    def from_dict(coords: dict[str, float], title: str = "") -> "BBox":
        assert isinstance(coords, dict) and set(coords) == {
            "x1",
            "y1",
            "x2",
            "y2",
        }, "Bounding box must be a dictionary with keys 'x1', 'y1', 'x2' and 'y2'."
        return BBox.from_list(
            [coords["x1"], coords["y1"], coords["x2"], coords["y2"]],
            title=title,
        )


class OBBox(DataModel):
    """
    A data model for representing oriented bounding boxes.

    Attributes:
        title (str): The title of the oriented bounding box.
        coords (list[int]): The coordinates of the oriented bounding box.

    The oriented bounding box is defined by four points:
        - (x1, y1): The first corner of the box.
        - (x2, y2): The second corner of the box.
        - (x3, y3): The third corner of the box.
        - (x4, y4): The fourth corner of the box.
    """

    title: str = Field(default="")
    coords: list[int] = Field(default=[])

    @staticmethod
    def from_list(coords: list[float], title: str = "") -> "OBBox":
        assert len(coords) == 8, (
            "Oriented bounding box must be a list of 8 coordinates."
        )
        assert all(isinstance(value, (int, float)) for value in coords), (
            "Oriented bounding box coordinates must be floats or integers."
        )
        return OBBox(
            title=title,
            coords=[round(c) for c in coords],
        )

    @staticmethod
    def from_dict(coords: dict[str, float], title: str = "") -> "OBBox":
        assert isinstance(coords, dict) and set(coords) == {
            "x1",
            "y1",
            "x2",
            "y2",
            "x3",
            "y3",
            "x4",
            "y4",
        }, "Oriented bounding box must be a dictionary with coordinates."
        return OBBox.from_list(
            [
                coords["x1"],
                coords["y1"],
                coords["x2"],
                coords["y2"],
                coords["x3"],
                coords["y3"],
                coords["x4"],
                coords["y4"],
            ],
            title=title,
        )
