import numpy as np
import pytest

from datachain.model.pose import Pose, Pose3D

POSE_KEYPOINTS = (
    [5 * i for i in range(17)],
    [3 * i for i in reversed(range(17))],
)
POSE_KEYPOINTS_NORMALIZED = (
    [x / 100 for x in POSE_KEYPOINTS[0]],
    [y / 100 for y in POSE_KEYPOINTS[1]],
)

POSE3D_KEYPOINTS = (
    POSE_KEYPOINTS[0],
    POSE_KEYPOINTS[1],
    [0.05 * i for i in range(17)],
)
POSE3D_KEYPOINTS_NORMALIZED = (
    POSE_KEYPOINTS_NORMALIZED[0],
    POSE_KEYPOINTS_NORMALIZED[1],
    [0.05 * i for i in range(17)],
)


@pytest.mark.parametrize(
    "points,normalized_to",
    [
        [POSE_KEYPOINTS, None],
        [tuple(tuple(c) for c in POSE_KEYPOINTS), None],
        [POSE_KEYPOINTS_NORMALIZED, (100, 100)],
    ],
)
def test_pose_from_list(points, normalized_to):
    pose = Pose.from_list(points, normalized_to=normalized_to)
    assert pose.model_dump() == {
        "x": POSE_KEYPOINTS[0],
        "y": POSE_KEYPOINTS[1],
    }
    np.testing.assert_array_almost_equal(
        pose.to_normalized((100, 100)),
        POSE_KEYPOINTS_NORMALIZED,
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
        [[[], []], None],
        [[[], [], []], None],
        [[12, 12], None],
        [[POSE_KEYPOINTS[0], POSE_KEYPOINTS[1] + [0]], None],
        [
            [
                [p * 2 for p in POSE_KEYPOINTS_NORMALIZED[0]],
                POSE_KEYPOINTS_NORMALIZED[1],
            ],
            (100, 100),
        ],
    ],
)
def test_pose_from_list_errors(points, normalized_to):
    with pytest.raises(AssertionError):
        Pose.from_list(points, normalized_to=normalized_to)


def test_pose_to_normalized_errors():
    with pytest.raises(AssertionError):
        Pose.from_list(POSE_KEYPOINTS).to_normalized((50, 50))


def test_pose_from_dict():
    pose = Pose.from_dict({"x": POSE_KEYPOINTS[0], "y": POSE_KEYPOINTS[1]})
    assert pose.model_dump() == {
        "x": POSE_KEYPOINTS[0],
        "y": POSE_KEYPOINTS[1],
    }


@pytest.mark.parametrize(
    "points",
    [
        {"x": POSE_KEYPOINTS[0]},
        {"x": POSE_KEYPOINTS[0], "y": POSE_KEYPOINTS[1], "z": []},
    ],
)
def test_pose_from_dict_errors(points):
    with pytest.raises(AssertionError):
        Pose.from_dict(points)


@pytest.mark.parametrize(
    "points,normalized_to",
    [
        [POSE3D_KEYPOINTS, None],
        [tuple(tuple(c) for c in POSE3D_KEYPOINTS), None],
        [POSE3D_KEYPOINTS_NORMALIZED, (100, 100)],
    ],
)
def test_pose3d_from_list(points, normalized_to):
    pose = Pose3D.from_list(points, normalized_to=normalized_to)
    assert pose.model_dump() == {
        "x": POSE3D_KEYPOINTS[0],
        "y": POSE3D_KEYPOINTS[1],
        "visible": POSE3D_KEYPOINTS[2],
    }
    np.testing.assert_array_almost_equal(
        pose.to_normalized((100, 100)),
        POSE3D_KEYPOINTS_NORMALIZED,
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
        [[[], []], None],
        [[[], [], []], None],
        [[12, 12], None],
        [[POSE3D_KEYPOINTS[0], POSE3D_KEYPOINTS[1] + [0]], None],
        [
            [
                [p * 2 for p in POSE3D_KEYPOINTS_NORMALIZED[0]],
                POSE3D_KEYPOINTS_NORMALIZED[1],
            ],
            (100, 100),
        ],
    ],
)
def test_pose3d_from_list_errors(points, normalized_to):
    with pytest.raises(AssertionError):
        Pose3D.from_list(points, normalized_to=normalized_to)


def test_pose3d_to_normalized_errors():
    with pytest.raises(AssertionError):
        Pose3D.from_list(POSE3D_KEYPOINTS).to_normalized((50, 50))


def test_pose3d_from_dict():
    pose = Pose3D.from_dict(
        {
            "x": POSE3D_KEYPOINTS[0],
            "y": POSE3D_KEYPOINTS[1],
            "visible": POSE3D_KEYPOINTS[2],
        }
    )
    assert pose.model_dump() == {
        "x": POSE3D_KEYPOINTS[0],
        "y": POSE3D_KEYPOINTS[1],
        "visible": POSE3D_KEYPOINTS[2],
    }


@pytest.mark.parametrize(
    "points",
    [
        {"x": POSE_KEYPOINTS[0], "y": POSE_KEYPOINTS[1]},
        {
            "x": POSE_KEYPOINTS[0],
            "y": POSE_KEYPOINTS[1],
            "visible": POSE3D_KEYPOINTS[2],
            "z": [],
        },
    ],
)
def test_pose3d_from_dict_errors(points):
    with pytest.raises(AssertionError):
        Pose3D.from_dict(points)
