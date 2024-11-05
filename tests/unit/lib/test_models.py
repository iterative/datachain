from datachain.lib import models


def test_bbox():
    bbox = models.BBox(x1=0.5, y1=1.5, x2=2.5, y2=3.5)
    assert bbox.model_dump() == {"x1": 0.5, "y1": 1.5, "x2": 2.5, "y2": 3.5}


def test_bbox_from_xywh():
    bbox = models.BBox.from_xywh([0.5, 1.5, 2.5, 3.5])
    assert bbox.model_dump() == {"x1": 0.5, "y1": 1.5, "x2": 3, "y2": 5}


def test_pose():
    x = list(x * 0.5 for x in range(17))
    y = list(y * 1.5 for y in range(17))
    pose = models.Pose(x=x, y=y)
    assert pose.model_dump() == {"x": x, "y": y}
    assert pose.x[models.yolo.PoseBodyPart.nose] == 0
    assert pose.x[models.yolo.PoseBodyPart.left_eye] == 0.5
    assert pose.x[models.yolo.PoseBodyPart.right_eye] == 1
    assert pose.x[models.yolo.PoseBodyPart.left_ear] == 1.5
    assert pose.x[models.yolo.PoseBodyPart.right_ear] == 2
    assert pose.x[models.yolo.PoseBodyPart.left_shoulder] == 2.5
    assert pose.x[models.yolo.PoseBodyPart.right_shoulder] == 3
    assert pose.x[models.yolo.PoseBodyPart.left_elbow] == 3.5
    assert pose.x[models.yolo.PoseBodyPart.right_elbow] == 4
    assert pose.x[models.yolo.PoseBodyPart.left_wrist] == 4.5
    assert pose.x[models.yolo.PoseBodyPart.right_wrist] == 5
    assert pose.x[models.yolo.PoseBodyPart.left_hip] == 5.5
    assert pose.x[models.yolo.PoseBodyPart.right_hip] == 6
    assert pose.x[models.yolo.PoseBodyPart.left_knee] == 6.5
    assert pose.x[models.yolo.PoseBodyPart.right_knee] == 7
    assert pose.x[models.yolo.PoseBodyPart.left_ankle] == 7.5
    assert pose.x[models.yolo.PoseBodyPart.right_ankle] == 8
