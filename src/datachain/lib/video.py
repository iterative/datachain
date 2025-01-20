import os.path
import pathlib
from typing import TYPE_CHECKING, Optional, Union

from datachain.lib.file import Video

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy import ndarray

    from datachain.lib.file import VideoFile

try:
    import imageio.v3 as iio
    from moviepy.video.io.VideoFileClip import VideoFileClip
except ImportError as exc:
    raise ImportError(
        "Missing dependencies for processing video.\n"
        "To install run:\n\n"
        "  pip install 'datachain[video]'\n"
    ) from exc


def video_meta(file: "VideoFile") -> Video:
    """
    Returns video file meta information.

    Args:
        file (VideoFile): VideoFile object.

    Returns:
        Video: Video file meta information.
    """
    props = iio.improps(file.stream(), plugin="pyav")
    frames, width, height, _ = props.shape

    meta = iio.immeta(file.stream(), plugin="pyav")
    fps = meta.get("fps", 0)
    duration = meta.get("duration", 0)
    format = meta.get("video_format", "")
    codec = meta.get("codec", "")

    return Video(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frames=frames,
        format=format,
        codec=codec,
    )


def video_frame_np(file: "VideoFile", frame: int) -> "ndarray":
    """
    Reads video frame from a file.

    Args:
        file (VideoFile): VideoFile object.
        frame (int): Frame number to read.

    Returns:
        ndarray: Video frame.
    """
    if frame < 0:
        raise ValueError("frame must be a non-negative integer.")

    return iio.imread(file.stream(), index=frame, plugin="pyav")


def video_frame(file: "VideoFile", frame: int, format: str = "jpeg") -> bytes:
    """
    Reads video frame from a file and returns as image bytes.

    Args:
        file (VideoFile): VideoFile object.
        frame (int): Frame number to read.
        format (str): Image format (default: 'jpeg').

    Returns:
        bytes: Video frame image as bytes.
    """
    img = video_frame_np(file, frame)
    return iio.imwrite("<bytes>", img, extension=f".{format}")


def save_video_frame(
    file: "VideoFile",
    frame: int,
    output_file: Union[str, pathlib.Path],
    format: str = "jpeg",
) -> None:
    """
    Saves video frame as an image file.

    Args:
        file (VideoFile): VideoFile object.
        frame (int): Frame number to read.
        output_file (Union[str, pathlib.Path]): Output file path.
        format (str): Image format (default: 'jpeg').
    """
    img = video_frame_np(file, frame)
    iio.imwrite(output_file, img, extension=f".{format}")


def video_frames_np(
    file: "VideoFile",
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
) -> "Iterator[ndarray]":
    """
    Reads video frames from a file.

    Args:
        file (VideoFile): VideoFile object.
        start_frame (int): Frame number to start reading from (default: 0).
        end_frame (int): Frame number to stop reading at (default: None).
        step (int): Step size for reading frames (default: 1).

    Returns:
        Iterator[ndarray]: Iterator of video frames.
    """
    if start_frame < 0:
        raise ValueError("start_frame must be a non-negative integer.")
    if end_frame is not None:
        if end_frame < 0:
            raise ValueError("end_frame must be a non-negative integer.")
        if start_frame > end_frame:
            raise ValueError("start_frame must be less than or equal to end_frame.")
    if step < 1:
        raise ValueError("step must be a positive integer.")

    # Compute the frame shift to determine the number of frames to skip,
    # considering the start frame and step size
    frame_shift = start_frame % step

    # Iterate over video frames and yield only those within the specified range and step
    for frame, img in enumerate(iio.imiter(file.stream(), plugin="pyav")):
        if frame < start_frame:
            continue
        if (frame - frame_shift) % step != 0:
            continue
        if end_frame is not None and frame > end_frame:
            break
        yield img


def video_frames(
    file: "VideoFile",
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    format: str = "jpeg",
) -> "Iterator[bytes]":
    """
    Reads video frames from a file and returns as bytes.

    Args:
        file (VideoFile): VideoFile object.
        start_frame (int): Frame number to start reading from (default: 0).
        end_frame (int): Frame number to stop reading at (default: None).
        step (int): Step size for reading frames (default: 1).
        format (str): Image format (default: 'jpeg').

    Returns:
        Iterator[bytes]: Iterator of video frames.
    """
    for img in video_frames_np(file, start_frame, end_frame, step):
        yield iio.imwrite("<bytes>", img, extension=f".{format}")


def save_video_frames(
    file: "VideoFile",
    output_dir: Union[str, pathlib.Path],
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    format: str = "jpeg",
) -> "Iterator[str]":
    """
    Saves video frames as image files.

    Args:
        file (VideoFile): VideoFile object.
        output_dir (Union[str, pathlib.Path]): Output directory path.
        start_frame (int): Frame number to start reading from (default: 0).
        end_frame (int): Frame number to stop reading at (default: None).
        step (int): Step size for reading frames (default: 1).
        format (str): Image format (default: 'jpeg').

    Returns:
        Iterator[str]: List of output file paths.
    """
    file_stem = file.get_file_stem()

    for i, img in enumerate(video_frames_np(file, start_frame, end_frame, step)):
        frame = start_frame + i * step
        output_file = os.path.join(output_dir, f"{file_stem}_{frame:06d}.{format}")
        iio.imwrite(output_file, img, extension=f".{format}")
        yield output_file


def save_video_clip(
    file: "VideoFile",
    start_time: float,
    end_time: float,
    output_file: Union[str, pathlib.Path],
    codec: str = "libx264",
    audio_codec: str = "aac",
) -> None:
    """
    Saves video interval as a new video file.

    Args:
        file (VideoFile): VideoFile object.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        output_file (Union[str, pathlib.Path]): Output file path.
        codec (str): Video codec for encoding (default: 'libx264').
        audio_codec (str): Audio codec for encoding (default: 'aac').
    """
    video = VideoFileClip(file.get_local_path())

    if start_time < 0 or end_time > video.duration or start_time >= end_time:
        raise ValueError(f"Invalid time range: ({start_time}, {end_time}).")

    clip = video.subclip(start_time, end_time)
    clip.write_videofile(output_file, codec=codec, audio_codec=audio_codec)
    video.close()


def save_video_clips(
    file: "VideoFile",
    intervals: list[tuple[float, float]],
    output_dir: Union[str, pathlib.Path],
    codec: str = "libx264",
    audio_codec: str = "aac",
) -> "Iterator[str]":
    """
    Saves video interval as a new video file.

    Args:
        file (VideoFile): VideoFile object.
        intervals (list[tuple[float, float]]): List of start and end times in seconds.
        output_dir (Union[str, pathlib.Path]): Output directory path.
        codec (str): Video codec for encoding (default: 'libx264').
        audio_codec (str): Audio codec for encoding (default: 'aac').

    Returns:
        Iterator[str]: List of output file paths.
    """
    file_stem = file.get_file_stem()
    file_ext = file.get_file_ext()

    video = VideoFileClip(file.stream())

    for i, (start, end) in enumerate(intervals):
        if start < 0 or end > video.duration or start >= end:
            print(f"Invalid time range: ({start}, {end}). Skipping this segment.")
            continue

        # Extract the segment
        clip = video.subclip(start, end)

        # Define the output file name
        output_file = os.path.join(output_dir, f"{file_stem}_{i + 1}.{file_ext}")

        # Write the video segment to file
        clip.write_videofile(output_file, codec=codec, audio_codec=audio_codec)

        yield output_file

    video.close()
