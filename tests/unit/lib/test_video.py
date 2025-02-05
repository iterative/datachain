import io
import os
import posixpath

import pytest
from numpy import ndarray
from PIL import Image

from datachain.lib.file import FileError, VideoFile


@pytest.fixture(autouse=True)
def video_file(catalog) -> VideoFile:
    data_path = os.path.join(os.path.dirname(__file__), "data")
    file_name = "Big_Buck_Bunny_360_10s_1MB.mp4"

    with open(os.path.join(data_path, file_name), "rb") as f:
        file = VideoFile.upload(f.read(), file_name)

    file.ensure_cached()
    return file


def test_get_info(video_file):
    info = video_file.get_info()
    assert info.model_dump() == {
        "width": 640,
        "height": 360,
        "fps": 30.0,
        "duration": 10.0,
        "frames": 300,
        "format": "mov,mp4,m4a,3gp,3g2,mj2",
        "codec": "h264",
    }


def test_get_info_error():
    # upload current Python file as video file to get an error while getting video meta
    with open(__file__, "rb") as f:
        file = VideoFile.upload(f.read(), "test.mp4")

    file.ensure_cached()
    with pytest.raises(FileError):
        file.get_info()


def test_get_frame_np(video_file):
    frame = video_file.get_frame_np(0)
    assert isinstance(frame, ndarray)
    assert frame.shape == (360, 640, 3)


def test_get_frame_np_error(video_file):
    with pytest.raises(ValueError):
        video_file.get_frame_np(-1)


@pytest.mark.parametrize(
    "format,img_format,header",
    [
        ("jpg", "JPEG", [b"\xff\xd8\xff\xe0"]),
        ("png", "PNG", [b"\x89PNG\r\n\x1a\n"]),
        ("gif", "GIF", [b"GIF87a", b"GIF89a"]),
    ],
)
def test_get_frame(video_file, format, img_format, header):
    frame = video_file.get_frame(0, format=format)
    assert isinstance(frame, bytes)
    assert any(frame.startswith(h) for h in header)

    img = Image.open(io.BytesIO(frame))
    assert img.format == img_format
    assert img.size == (640, 360)


@pytest.mark.parametrize("use_format", [True, False])
def test_save_frame_ext(tmp_path, video_file, use_format):
    filename = "frame" if use_format else "frame.jpg"
    format = "jpg" if use_format else None
    output_file = posixpath.join(tmp_path, filename)

    frame_file = video_file.save_frame(3, str(output_file), format=format)
    assert frame_file.frame == 3
    assert frame_file.timestamp == 3 / 30

    frame_file.ensure_cached()
    img = Image.open(frame_file.get_local_path())
    assert img.format == "JPEG"
    assert img.size == (640, 360)


def test_get_frames_np(video_file):
    frames = list(video_file.get_frames_np(10, 200, 5))
    assert len(frames) == 39
    assert all(isinstance(frame, ndarray) for frame in frames)
    assert all(frame.shape == (360, 640, 3) for frame in frames)


@pytest.mark.parametrize(
    "start_frame,end_frame,step",
    [
        (-1, None, None),
        (0, -1, None),
        (1, 0, None),
        (0, 1, -1),
    ],
)
def test_get_frames_np_error(video_file, start_frame, end_frame, step):
    with pytest.raises(ValueError):
        list(video_file.get_frames_np(start_frame, end_frame, step))


def test_get_frames(video_file):
    frames = list(video_file.get_frames(10, 200, 5, format="jpg"))
    assert len(frames) == 39
    assert all(isinstance(frame, bytes) for frame in frames)
    assert all(Image.open(io.BytesIO(frame)).format == "JPEG" for frame in frames)


def test_save_frames(tmp_path, video_file):
    frame_files = list(video_file.save_frames(str(tmp_path), 10, 200, 5, format="jpg"))
    assert len(frame_files) == 39

    for i, frame_file in enumerate(frame_files):
        assert frame_file.frame == 10 + 5 * i
        assert frame_file.timestamp == (10 + 5 * i) / 30

        frame_file.ensure_cached()
        img = Image.open(frame_file.get_local_path())
        assert img.format == "JPEG"
        assert img.size == (640, 360)


def test_save_fragment(tmp_path, video_file):
    output_file = posixpath.join(tmp_path, "fragment.mp4")
    fragment = video_file.save_fragment(2.5, 5, str(output_file))
    assert fragment.start == 2.5
    assert fragment.end == 5

    fragment.ensure_cached()
    assert fragment.get_info().model_dump() == {
        "width": 640,
        "height": 360,
        "fps": 30.0,
        "duration": 2.5,
        "frames": 75,
        "format": "mov,mp4,m4a,3gp,3g2,mj2",
        "codec": "h264",
    }


def test_save_fragment_error(video_file):
    with pytest.raises(ValueError):
        video_file.save_fragment(5, 2.5, "fragment.mp4")


def test_save_fragments(tmp_path, video_file):
    intervals = [(1, 2), (3, 4), (5, 6)]

    fragments = list(video_file.save_fragments(intervals, str(tmp_path)))
    assert len(fragments) == 3

    for i, fragment in enumerate(fragments):
        assert fragment.start == 1 + 2 * i
        assert fragment.end == 2 + 2 * i

        fragment.ensure_cached()
        assert fragment.get_info().model_dump() == {
            "width": 640,
            "height": 360,
            "fps": 30.0,
            "duration": 1,
            "frames": 30,
            "format": "mov,mp4,m4a,3gp,3g2,mj2",
            "codec": "h264",
        }


def test_save_fragments_error(video_file):
    fragments = list(video_file.save_fragments([(2, 1)], "fragments"))
    assert len(fragments) == 0
