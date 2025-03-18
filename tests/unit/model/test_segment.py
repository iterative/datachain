import pytest

from datachain import model


@pytest.mark.parametrize(
    "segments",
    [
        model.Segment(x=[0, 1, 2], y=[2, 3, 5], title="Segments"),
        model.Segment.from_list([[0, 1, 2], [2, 3, 5]], title="Segments"),
        model.Segment.from_dict({"x": [0, 1, 2], "y": [2, 3, 5]}, title="Segments"),
    ],
)
def test_segments(segments):
    assert segments.model_dump() == {
        "title": "Segments",
        "x": [0, 1, 2],
        "y": [2, 3, 5],
    }


@pytest.mark.parametrize(
    "points,exception",
    [
        (None, TypeError),
        ([], ValueError),
        ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], ValueError),
        (
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [0, 2, 4]],
            ValueError,
        ),
        ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], None], TypeError),
        (
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "11", 12, 13, 14, 15, 16],
                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
            ],
            ValueError,
        ),
    ],
)
def test_segments_from_list_error(points, exception):
    with pytest.raises(exception):
        model.Segment.from_list(points)


def test_segments_from_dict_error():
    with pytest.raises(ValueError):
        model.Segment.from_dict(
            {
                "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            }
        )
