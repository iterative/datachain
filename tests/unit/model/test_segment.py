from datachain.model.segment import Segment


def test_segment():
    segment = Segment(title="Object", x=[10, 20, 30], y=[40, 50, 60])
    assert segment.model_dump() == {
        "title": "Object",
        "x": [10.0, 20.0, 30.0],
        "y": [40.0, 50.0, 60.0],
    }


def test_segment_empty():
    assert Segment().model_dump() == {"title": "", "x": [], "y": []}
