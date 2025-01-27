import os.path
from typing import TYPE_CHECKING, Optional

from datachain.lib.file import File, FileError, Video, VideoFragment, VideoFrame

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy import ndarray

try:
    import ffmpeg
    import imageio.v3 as iio
except ImportError as exc:
    raise ImportError(
        "Missing dependencies for processing video.\n"
        "To install run:\n\n"
        "  pip install 'datachain[video]'\n"
    ) from exc


def _video_probe(file: "File") -> tuple[dict, dict, float]:
    """Probes video file for video stream, video format and fps."""
    try:
        probe = ffmpeg.probe(file.get_local_path())
    except ffmpeg.Error as exc:
        raise FileError(file, f"unable to probe video file: {exc.stderr}") from exc
    except Exception as exc:
        raise FileError(file, f"unable to probe video file: {exc}") from exc

    if not probe:
        raise FileError(file, "unable to probe video file")

    all_streams = probe.get("streams")
    video_format = probe.get("format")
    if not all_streams or not video_format:
        raise FileError(file, "unable to probe video file")

    video_streams = [s for s in all_streams if s["codec_type"] == "video"]
    if len(video_streams) == 0:
        raise FileError(file, "no video streams found in video file")

    video_stream = video_streams[0]

    r_frame_rate = video_stream.get("r_frame_rate", "0")
    if "/" in r_frame_rate:
        num, denom = r_frame_rate.split("/")
        fps = float(num) / float(denom)
    else:
        fps = float(r_frame_rate)

    return video_stream, video_format, fps


def _video_fps(file: "File") -> float:
    """Returns video file frames per second."""
    _, _, fps = _video_probe(file)
    return fps


def video_info(file: "File") -> "Video":
    """
    Returns video file information.

    Args:
        file (File): File object.

    Returns:
        Video: Video file information.
    """
    video_stream, video_format, fps = _video_probe(file)

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    duration = float(video_format.get("duration", 0))
    start_time = float(video_format.get("start_time", 0))
    frames = round((duration - start_time) * fps)
    format_name = video_format.get("format_name", "")
    codec_name = video_stream.get("codec_name", "")

    return Video(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frames=frames,
        format=format_name,
        codec=codec_name,
    )


def video_frame_np(file: "File", frame: int) -> "ndarray":
    """
    Reads video frame from a file.

    Args:
        file (File): File object.
        frame (int): Frame number to read.

    Returns:
        ndarray: Video frame.
    """
    if frame < 0:
        raise ValueError("frame must be a non-negative integer.")

    return iio.imread(file.stream(), index=frame, plugin="pyav")


def video_frame(file: "File", frame: int, format: str = "jpg") -> bytes:
    """
    Reads video frame from a file and returns as image bytes.

    Args:
        file (File): File object.
        frame (int): Frame number to read.
        format (str): Image format (default: 'jpg').

    Returns:
        bytes: Video frame image as bytes.
    """
    img = video_frame_np(file, frame)
    return iio.imwrite("<bytes>", img, extension=f".{format}")


def save_video_frame(
    file: "File",
    frame: int,
    output_file: str,
) -> "VideoFrame":
    """
    Saves video frame as an image file.

    Args:
        file (File): File object.
        frame (int): Frame number to read.
        output_file (str): Output file path.

    Returns:
        VideoFrame: Video frame model.
    """
    _, _, fps = _video_probe(file)

    ext = os.path.splitext(output_file)[1]

    img = video_frame_np(file, frame)
    iio.imwrite(output_file, img, extension=f".{ext}")

    frame_file = File.from_local_path(output_file)
    return VideoFrame(
        **frame_file.model_dump(),
        frame=frame,
        timestamp=float(frame) / fps,
        orig=file,
    )


def video_frames_np(
    file: "File",
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
) -> "Iterator[ndarray]":
    """
    Reads video frames from a file.

    Args:
        file (File): File object.
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
    file: "File",
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    format: str = "jpg",
) -> "Iterator[bytes]":
    """
    Reads video frames from a file and returns as bytes.

    Args:
        file (File): File object.
        start_frame (int): Frame number to start reading from (default: 0).
        end_frame (int): Frame number to stop reading at (default: None).
        step (int): Step size for reading frames (default: 1).
        format (str): Image format (default: 'jpg').

    Returns:
        Iterator[bytes]: Iterator of video frames.
    """
    for img in video_frames_np(file, start_frame, end_frame, step):
        yield iio.imwrite("<bytes>", img, extension=f".{format}")


def save_video_frames(
    file: "File",
    output_dir: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    format: str = "jpg",
) -> "Iterator[VideoFrame]":
    """
    Saves video frames as image files.

    Args:
        file (File): File object.
        output_dir (str): Output directory path.
        start_frame (int): Frame number to start reading from (default: 0).
        end_frame (int): Frame number to stop reading at (default: None).
        step (int): Step size for reading frames (default: 1).
        format (str): Image format (default: 'jpg').

    Returns:
        Iterator[VideoFrame]: List of video frame models.
    """
    _, _, fps = _video_probe(file)
    file_stem = file.get_file_stem()

    for i, img in enumerate(video_frames_np(file, start_frame, end_frame, step)):
        frame = start_frame + i * step
        output_file = os.path.join(output_dir, f"{file_stem}_{frame:06d}.{format}")

        iio.imwrite(output_file, img, extension=f".{format}")

        frame_file = File.from_local_path(output_file)
        yield VideoFrame(
            **frame_file.model_dump(),
            frame=frame,
            timestamp=frame / fps,
            orig=file,
        )


def save_video_fragment(
    file: "File",
    start_time: float,
    end_time: float,
    output_file: str,
) -> "VideoFragment":
    """
    Saves video interval as a new video file.

    Args:
        file (File): File object.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        output_file (str): Output file path.

    Returns:
        VideoFragment: Video fragment model.
    """
    if start_time < 0 or start_time >= end_time:
        raise ValueError(f"Invalid time range: ({start_time}, {end_time}).")

    (
        ffmpeg.input(file.get_local_path(), ss=start_time, to=end_time)
        .output(output_file)
        .run(quiet=True)
    )

    clip_file = File.from_local_path(output_file)
    return VideoFragment(
        **clip_file.model_dump(),
        start=start_time,
        end=end_time,
        orig=file,
    )


def save_video_fragments(
    file: "File",
    intervals: list[tuple[float, float]],
    output_dir: str,
) -> "Iterator[VideoFragment]":
    """
    Saves video intervals as new video files.

    Args:
        file (File): File object.
        intervals (list[tuple[float, float]]): List of start and end times in seconds.
        output_dir (str): Output directory path.

    Returns:
        Iterator[VideoFragment]: List of video fragment models.
    """
    file_stem = file.get_file_stem()
    file_ext = file.get_file_ext()

    for i, (start, end) in enumerate(intervals):
        if start < 0 or start >= end:
            print(f"Invalid time range: ({start}, {end}). Skipping this segment.")
            continue

        # Define the output file name
        output_file = os.path.join(output_dir, f"{file_stem}_{i + 1}.{file_ext}")

        # Write the video fragment to file and yield it
        yield save_video_fragment(file, start, end, output_file)
