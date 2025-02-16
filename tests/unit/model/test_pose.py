from datachain.model.pose import Pose, Pose3D


def test_pose():
    pose = Pose(x=[10, 20, 30], y=[40, 50, 60])
    assert pose.model_dump() == {
        "x": [10.0, 20.0, 30.0],
        "y": [40.0, 50.0, 60.0],
    }


def test_pose_empty():
    assert Pose().model_dump() == {"x": [], "y": []}


def test_pose3d():
    pose3d = Pose3D(x=[10, 20, 30], y=[40, 50, 60], visible=[0.0, 0.5, 1.0])
    assert pose3d.model_dump() == {
        "x": [10.0, 20.0, 30.0],
        "y": [40.0, 50.0, 60.0],
        "visible": [0.0, 0.5, 1.0],
    }


def test_pose3d_empty():
    assert Pose3D().model_dump() == {"x": [], "y": [], "visible": []}
