from numpy.testing import assert_array_almost_equal

from datachain.model.yolo import (
    YoloBox,
    YoloCls,
    YoloObb,
    YoloPose,
    YoloPoseBodyPart,
    YoloSeg,
)


def test_yolo_box():
    model = YoloBox(
        cls=[1, 2],
        name=["person", "chair"],
        confidence=[0.8, 0.5],
        box=[
            [10, 20, 190, 80],
            [160, 80, 200, 100],
        ],
        orig_shape=[100, 200],
    )

    assert model.model_dump() == {
        "cls": [1, 2],
        "name": ["person", "chair"],
        "confidence": [0.8, 0.5],
        "box": [
            [10, 20, 190, 80],
            [160, 80, 200, 100],
        ],
        "orig_shape": [100, 200],
    }

    assert model.img_size == (200, 100)

    alb_boxes = list(model.to_albumentations())
    assert len(alb_boxes) == 2
    assert_array_almost_equal(alb_boxes[0], [0.05, 0.2, 0.95, 0.8], decimal=3)
    assert_array_almost_equal(alb_boxes[1], [0.8, 0.8, 1.0, 1.0], decimal=3)

    coco_boxes = list(model.to_coco())
    assert len(coco_boxes) == 2
    assert coco_boxes[0] == [10, 20, 180, 60]
    assert coco_boxes[1] == [160, 80, 40, 20]

    voc_boxes = list(model.to_voc())
    assert len(voc_boxes) == 2
    assert voc_boxes[0] == [10, 20, 190, 80]
    assert voc_boxes[1] == [160, 80, 200, 100]

    yolo_boxes = list(model.to_yolo())
    assert len(yolo_boxes) == 2
    assert_array_almost_equal(yolo_boxes[0], [0.5, 0.5, 0.9, 0.6], decimal=3)
    assert_array_almost_equal(yolo_boxes[1], [0.9, 0.9, 0.2, 0.2], decimal=3)


def test_yolo_obb():
    model = YoloObb(
        cls=[1, 2],
        name=["person", "chair"],
        confidence=[0.8, 0.5],
        obox=[
            [10, 20, 30, 40, 50, 60, 70, 80],
            [110, 120, 130, 140, 150, 160, 170, 180],
        ],
        orig_shape=[100, 200],
    )

    assert model.model_dump() == {
        "cls": [1, 2],
        "name": ["person", "chair"],
        "confidence": [0.8, 0.5],
        "obox": [
            [10, 20, 30, 40, 50, 60, 70, 80],
            [110, 120, 130, 140, 150, 160, 170, 180],
        ],
        "orig_shape": [100, 200],
    }

    assert model.img_size == (200, 100)


def test_yolo_seg():
    model = YoloSeg(
        cls=[1, 2],
        name=["person", "chair"],
        confidence=[0.8, 0.5],
        box=[
            [10, 20, 190, 80],
            [160, 80, 200, 100],
        ],
        segments=[
            [[10, 20], [30, 40], [50, 60]],
            [[110, 120], [130, 140], [150, 160]],
        ],
        orig_shape=[100, 200],
    )

    assert model.model_dump() == {
        "cls": [1, 2],
        "name": ["person", "chair"],
        "confidence": [0.8, 0.5],
        "box": [
            [10, 20, 190, 80],
            [160, 80, 200, 100],
        ],
        "segments": [
            [[10, 20], [30, 40], [50, 60]],
            [[110, 120], [130, 140], [150, 160]],
        ],
        "orig_shape": [100, 200],
    }

    assert model.img_size == (200, 100)


def test_yolo_pose():
    model = YoloPose(
        cls=[1],
        name=["person"],
        confidence=[0.8],
        box=[
            [10, 20, 190, 80],
        ],
        keypoints=[
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
                [0.5 for _ in range(17)],
            ],
        ],
        orig_shape=[100, 200],
    )

    assert model.model_dump() == {
        "cls": [1],
        "name": ["person"],
        "confidence": [0.8],
        "box": [
            [10, 20, 190, 80],
        ],
        "keypoints": [
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
                [
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                ],
            ],
        ],
        "orig_shape": [100, 200],
    }

    assert model.img_size == (200, 100)

    assert model.keypoints[0][0][YoloPoseBodyPart.left_wrist] == 10
    assert model.keypoints[0][1][YoloPoseBodyPart.left_wrist] == 20
    assert model.keypoints[0][2][YoloPoseBodyPart.left_wrist] == 0.5


def test_yolo_cls():
    model = YoloCls(
        cls=[1, 37],
        name=["person", "chair"],
        confidence=[0.8, 0.6],
    )

    assert model.model_dump() == {
        "cls": [1, 37],
        "name": ["person", "chair"],
        "confidence": [0.8, 0.6],
    }
