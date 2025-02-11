import numpy as np
import pytest

from datachain.model.segment import Segment

SEGMENT_POINTS = (
    [2 * i for i in range(50)],
    list(reversed(range(50))),
)
SEGMENT_POINTS_NORMALIZED = (
    [x / 100 for x in SEGMENT_POINTS[0]],
    [y / 100 for y in SEGMENT_POINTS[1]],
)


@pytest.mark.parametrize(
    "points,title,normalized_to",
    [
        [SEGMENT_POINTS, "Segment", None],
        [tuple(tuple(c) for c in SEGMENT_POINTS), "", None],
        [SEGMENT_POINTS_NORMALIZED, "Person", (100, 100)],
    ],
)
def test_pose_from_list(points, title, normalized_to):
    segment = Segment.from_list(points, title, normalized_to=normalized_to)
    assert segment.model_dump() == {
        "title": title,
        "x": SEGMENT_POINTS[0],
        "y": SEGMENT_POINTS[1],
    }
    np.testing.assert_array_almost_equal(
        segment.to_normalized((100, 100)),
        SEGMENT_POINTS_NORMALIZED,
    )


@pytest.mark.parametrize(
    "points,normalized_to",
    [
        [None, None],
        [12, None],
        ["12", None],
        [[], None],
        [[12, []], None],
        [[[], "12"], None],
        [[[], [], []], None],
        [[12, 12], None],
        [[SEGMENT_POINTS[0], SEGMENT_POINTS[1] + [0]], None],
        [
            [
                [p * 2 for p in SEGMENT_POINTS_NORMALIZED[0]],
                SEGMENT_POINTS_NORMALIZED[1],
            ],
            (100, 100),
        ],
    ],
)
def test_pose_from_list_errors(points, normalized_to):
    with pytest.raises(AssertionError):
        Segment.from_list(points, normalized_to=normalized_to)


def test_pose_to_normalized_errors():
    with pytest.raises(AssertionError):
        Segment.from_list(SEGMENT_POINTS).to_normalized((50, 50))


def test_pose_from_dict():
    segment = Segment.from_dict({"x": SEGMENT_POINTS[0], "y": SEGMENT_POINTS[1]})
    assert segment.model_dump() == {
        "title": "",
        "x": SEGMENT_POINTS[0],
        "y": SEGMENT_POINTS[1],
    }


@pytest.mark.parametrize(
    "points",
    [
        {"x": SEGMENT_POINTS[0]},
        {"x": SEGMENT_POINTS[0], "y": SEGMENT_POINTS[1], "z": []},
    ],
)
def test_pose_from_dict_errors(points):
    with pytest.raises(AssertionError):
        Segment.from_dict(points)
