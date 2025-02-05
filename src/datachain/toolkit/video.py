from collections.abc import Iterator
from typing import Callable, Optional

from datachain.lib.video import VideoFile, VideoFrame


def split_video_to_frames(
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1,
) -> Callable[[VideoFile], Iterator[VideoFrame]]:
    """
    Returns a UDF function that splits a video file into frames.

    Args:
        start (int, optional): The starting frame number (default: 0).
        end (int, optional): The ending frame number (exclusive). If None,
                             frames are read until the end of the video
                             (default: None).
        step (int, optional): The interval between frames to read (default: 1).

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


def split_video_to_fragments(
    duration: float,
    start: float = 0,
    end: Optional[float] = None,
) -> Callable[[VideoFile], Iterator[VideoFile]]:
    """
    Returns a UDF function that splits a video file into fragments.

    Args:
        duration (float): The duration of each video fragment in seconds.
        start (float, optional): The starting time in seconds (default: 0).
        end (float, optional): The ending time in seconds. If None, the entire
                               remaining video is processed (default: None).

    Returns:
        Callable[[VideoFile], Iterator[VideoFile]]: UDF function with VideoFile input
                                                    and VideoFile iterator output.

    Usage:
        Split videos from the "videos" dataset into fragments and save them
        to the "fragments" dataset:

        ```python
        from datachain import DataChain
        from datachain.toolkit.video import split_video_to_fragments

        (
            DataChain.from_dataset("videos")
                .gen(fragment=split_video_to_fragments(duration=1.0))
                .save("fragments")
        )
        ```
    """

    def _split_fragments(file: VideoFile) -> Iterator[VideoFile]:
        yield from file.get_fragments(duration=duration, start=start, end=end)

    return _split_fragments
