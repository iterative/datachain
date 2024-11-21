import pytest

from datachain import model
from datachain.model.ultralytics.pose import YoloPoseBodyPart


@pytest.mark.parametrize(
    "bbox",
    [
        model.BBox(title="BBox", coords=[0, 1, 2, 3]),
        model.BBox.from_list([0.3, 1.1, 1.7, 3.4], title="BBox"),
        model.BBox.from_dict({"x1": 0, "y1": 0.8, "x2": 2.2, "y2": 2.9}, title="BBox"),
    ],
)
def test_bbox(bbox):
    assert bbox.model_dump() == {
        "title": "BBox",
        "coords": [0, 1, 2, 3],
    }


@pytest.mark.parametrize(
    "obbox",
    [
        model.OBBox(title="OBBox", coords=[0, 1, 2, 3, 4, 5, 6, 7]),
        model.OBBox.from_list([0.3, 1.1, 1.7, 3.4, 4.0, 4.9, 5.6, 7.0], title="OBBox"),
        model.OBBox.from_dict(
            {
                "x1": 0,
                "y1": 0.8,
                "x2": 2.2,
                "y2": 2.9,
                "x3": 3.9,
                "y3": 5.4,
                "x4": 6.0,
                "y4": 7.4,
            },
            title="OBBox",
        ),
    ],
)
def test_obbox(obbox):
    assert obbox.model_dump() == {
        "title": "OBBox",
        "coords": [0, 1, 2, 3, 4, 5, 6, 7],
    }


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
