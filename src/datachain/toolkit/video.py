from collections.abc import Iterator
from typing import Callable

from datachain.lib.video import VideoFile, VideoFrame


def split_video_to_frames(
    start=0,
    end=None,
    step=1,
) -> Callable[[VideoFile], Iterator[VideoFrame]]:
    """
    Returns a UDF function that splits a video file into frames.

    Args:
        start (int): Start frame index (inclusive, default: first frame).
        end (Optional[int]): End frame index (exclusive, default: last frame).
        step (int): Step between frames (default: 1).

    Returns:
        Callable[[VideoFile], Iterator[VideoFrame]]: UDF function with VideoFile input
                                                     and VideoFrame iterator output.

    Usage:
        Split videos from the "videos" dataset into frames and save them
        to the "frames" dataset:

        ```python
        from datachain import DataChain
        from datachain.toolkit.video import split_video_to_frames

        (
            DataChain.from_dataset("videos")
                .gen(frame=split_video_to_frames(step=5))
                .save("frames")
        )
        ```
    """

    def _split_frames(file: VideoFile) -> Iterator[VideoFrame]:
        yield from file.get_frames(start=start, end=end, step=step)

    return _split_frames
