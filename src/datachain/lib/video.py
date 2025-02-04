import posixpath
import shutil
import tempfile
from typing import Optional

from numpy import ndarray

from datachain.lib.file import (
    FileError,
    ImageFile,
    Video,
    VideoFile,
    VideoFragment,
    VideoFrame,
)

try:
    import ffmpeg
    import imageio.v3 as iio
except ImportError as exc:
    raise ImportError(
        "Missing dependencies for processing video.\n"
        "To install run:\n\n"
        "  pip install 'datachain[video]'\n"
    ) from exc


def video_info(file: VideoFile) -> Video:
    """
    Returns video file information.

    Args:
        file (VideoFile): Video file object.

    Returns:
        Video: Video file information.
    """
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

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    duration = float(video_format.get("duration", 0))
    if "nb_frames" in video_stream:
        frames = int(video_stream.get("nb_frames", 0))
    else:
        start_time = float(video_format.get("start_time", 0))
        frames = int((duration - start_time) * fps)
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


def video_frame_np(file: VideoFrame) -> ndarray:
    """
    Reads video frame from a file and returns as numpy array.

    Args:
        file (VideoFrame): VideoFrame file object.

    Returns:
        ndarray: Video frame.
    """
    if file.frame < 0:
        raise ValueError("frame must be a non-negative integer")

    with file.open() as f:
        return iio.imread(f, index=file.frame, plugin="pyav")  # type: ignore[arg-type]


def video_frame_bytes(file: VideoFrame, format: str = "jpg") -> bytes:
    """
    Reads video frame from a file and returns as image bytes.

    Args:
        file (VideoFrame): VideoFrame file object.
        format (str): Image format (default: 'jpg').

    Returns:
        bytes: Video frame image as bytes.
    """
    img = video_frame_np(file)
    return iio.imwrite("<bytes>", img, extension=f".{format}")


def save_video_frame(
    file: VideoFrame,
    output: str,
    format: str = "jpg",
) -> ImageFile:
    """
    Saves video frame as a new image file. If output is a remote path,
    the image file will be uploaded to the remote storage.

    Args:
        file (VideoFrame): VideoFrame file object.
        output (str): Output path, can be a local path or a remote path.
        format (str): Image format (default: 'jpg').

    Returns:
        ImageFile: Image file model.
    """
    img = video_frame_bytes(file, format=format)
    output_file = posixpath.join(
        output, f"{file.get_file_stem()}_{file.frame:04d}.{format}"
    )
    return ImageFile.upload(img, output_file)


def save_video_fragment(
    file: VideoFragment,
    output: str,
    format: Optional[str] = None,
) -> VideoFile:
    """
    Saves video interval as a new video file. If output is a remote path,
    the video file will be uploaded to the remote storage.

    Args:
        file (VideoFragment): VideoFragment file object.
        output (str): Output path, can be a local path or a remote path.
        format (Optional[str]): Output format (default: None). If not provided,
                                the format will be inferred from the video fragment
                                file extension.

    Returns:
        VideoFile: Video fragment model.
    """
    if file.start < 0 or file.end < 0 or file.start >= file.end:
        raise ValueError(f"Invalid time range: ({file.start:.3f}, {file.end:.3f})")

    if format is None:
        format = file.get_file_ext()

    start_ms = int(file.start * 1000)
    end_ms = int(file.end * 1000)
    output_file = posixpath.join(
        output, f"{file.get_file_stem()}_{start_ms:06d}_{end_ms:06d}.{format}"
    )

    temp_dir = tempfile.mkdtemp()
    try:
        output_file_tmp = posixpath.join(temp_dir, posixpath.basename(output_file))
        ffmpeg.input(file.get_local_path(), ss=file.start, to=file.end).output(
            output_file_tmp
        ).run(quiet=True)

        with open(output_file_tmp, "rb") as f:
            return VideoFile.upload(f.read(), output_file)
    finally:
        shutil.rmtree(temp_dir)
