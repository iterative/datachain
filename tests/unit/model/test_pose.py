import pytest

from datachain import model
from datachain.model.ultralytics.pose import YoloPoseBodyPart


@pytest.mark.parametrize(
    "pose",
    [
        model.Pose(x=list(range(17)), y=[y * 2 for y in range(17)]),
        model.Pose.from_list([list(range(17)), [y * 2 for y in range(17)]]),
        model.Pose.from_dict({"x": list(range(17)), "y": [y * 2 for y in range(17)]}),
    ],
)
def test_pose(pose):
    assert pose.model_dump() == {
        "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "y": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
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
def test_pose_from_list_error(points, exception):
    with pytest.raises(exception):
        model.Pose.from_list(points)


def test_pose_from_dict_error():
    with pytest.raises(ValueError):
        model.Pose.from_dict(
            {
                "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            }
        )


@pytest.mark.parametrize(
    "pose",
    [
        model.Pose3D(
            x=list(range(17)), y=[y * 2 for y in range(17)], visible=[0.2] * 17
        ),
        model.Pose3D.from_list(
            [list(range(17)), [y * 2 for y in range(17)], [0.2] * 17]
        ),
        model.Pose3D.from_dict(
            {
                "x": list(range(17)),
                "y": [y * 2 for y in range(17)],
                "visible": [0.2] * 17,
            }
        ),
    ],
)
def test_pose3d(pose):
    assert pose.model_dump() == {
        "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "y": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
        "visible": [
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ],
    }


@pytest.mark.parametrize(
    "points,exception",
    [
        (None, TypeError),
        ([], ValueError),
        ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], ValueError),
        (
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [0, 2, 4], []],
            ValueError,
        ),
        (
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                [0, 2, 4],
                None,
            ],
            TypeError,
        ),
        (
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "11", 12, 13, 14, 15, 16],
                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
                [0.2] * 17,
            ],
            ValueError,
        ),
    ],
)
def test_pose3d_from_list_error(points, exception):
    with pytest.raises(exception):
        model.Pose3D.from_list(points)


def test_pose3d_from_dict_error():
    with pytest.raises(ValueError):
        model.Pose3D.from_dict(
            {
                "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                "y": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
                "visible": [0.2] * 16,
                "foo": "bar",
            }
        )


def test_yolo_pose_body_parts():
    pose = model.Pose(x=list(range(17)), y=list(range(17)))
    assert pose.x[YoloPoseBodyPart.nose] == 0
    assert pose.x[YoloPoseBodyPart.left_eye] == 1
    assert pose.x[YoloPoseBodyPart.right_eye] == 2
    assert pose.x[YoloPoseBodyPart.left_ear] == 3
    assert pose.x[YoloPoseBodyPart.right_ear] == 4
    assert pose.x[YoloPoseBodyPart.left_shoulder] == 5
    assert pose.x[YoloPoseBodyPart.right_shoulder] == 6
    assert pose.x[YoloPoseBodyPart.left_elbow] == 7
    assert pose.x[YoloPoseBodyPart.right_elbow] == 8
    assert pose.x[YoloPoseBodyPart.left_wrist] == 9
    assert pose.x[YoloPoseBodyPart.right_wrist] == 10
    assert pose.x[YoloPoseBodyPart.left_hip] == 11
    assert pose.x[YoloPoseBodyPart.right_hip] == 12
    assert pose.x[YoloPoseBodyPart.left_knee] == 13
    assert pose.x[YoloPoseBodyPart.right_knee] == 14
    assert pose.x[YoloPoseBodyPart.left_ankle] == 15
    assert pose.x[YoloPoseBodyPart.right_ankle] == 16
