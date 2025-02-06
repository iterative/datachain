from typing import Union

import numpy as np
import torch
from PIL import Image
from ultralytics.engine.results import Results

from datachain.model.ultralytics.bbox import YoloBBox, YoloBBoxes
from datachain.model.ultralytics.pose import YoloPose, YoloPoses
from datachain.model.ultralytics.segment import YoloSegment, YoloSegments

YoloSignal = Union[YoloBBox, YoloBBoxes, YoloPose, YoloPoses, YoloSegment, YoloSegments]


def _signal_to_results(img: np.ndarray, signal: YoloSignal) -> Results:
    # Convert RGB to BGR
    if img.ndim == 3 and img.shape[2] == 3:
        bgr_array = img[:, :, ::-1]
    else:
        # If the image is not RGB (e.g., grayscale or RGBA), use as is
        bgr_array = img

    names = {}
    boxes_list = []
    keypoints_list = []
    masks_list = []

    # Get the boxes, keypoints, and masks from the signal
    if isinstance(signal, YoloBBox):
        names[signal.cls] = signal.name
        boxes_list.append(
            torch.tensor([[*signal.box.coords, signal.confidence, signal.cls]])
        )
    elif isinstance(signal, YoloBBoxes):
        for i, _ in enumerate(signal.cls):
            names[signal.cls[i]] = signal.name[i]
            boxes_list.append(
                torch.tensor(
                    [[*signal.box[i].coords, signal.confidence[i], signal.cls[i]]]
                )
            )
    elif isinstance(signal, YoloPose):
        names[signal.cls] = signal.name
        boxes_list.append(
            torch.tensor([[*signal.box.coords, signal.confidence, signal.cls]])
        )
        keypoints_list.append(
            torch.tensor([list(zip(signal.pose.x, signal.pose.y, signal.pose.visible))])
        )
    elif isinstance(signal, YoloPoses):
        for i, _ in enumerate(signal.cls):
            names[signal.cls[i]] = signal.name[i]
            boxes_list.append(
                torch.tensor(
                    [[*signal.box[i].coords, signal.confidence[i], signal.cls[i]]]
                )
            )
            keypoints_list.append(
                torch.tensor(
                    [
                        list(
                            zip(
                                signal.pose[i].x,
                                signal.pose[i].y,
                                signal.pose[i].visible,
                            )
                        )
                    ]
                )
            )
    elif isinstance(signal, YoloSegment):
        names[signal.cls] = signal.name
        boxes_list.append(
            torch.tensor([[*signal.box.coords, signal.confidence, signal.cls]])
        )
        masks_list.append(torch.tensor([list(zip(signal.segment.x, signal.segment.y))]))
    elif isinstance(signal, YoloSegments):
        for i, _ in enumerate(signal.cls):
            names[signal.cls[i]] = signal.name[i]
            boxes_list.append(
                torch.tensor(
                    [[*signal.box[i].coords, signal.confidence[i], signal.cls[i]]]
                )
            )
            masks_list.append(
                torch.tensor([list(zip(signal.segment[i].x, signal.segment[i].y))])
            )

    boxes = torch.cat(boxes_list, dim=0) if len(boxes_list) > 0 else None
    keypoints = torch.cat(keypoints_list, dim=0) if len(keypoints_list) > 0 else None
    masks = torch.cat(masks_list, dim=0) if len(masks_list) > 0 else None

    return Results(
        bgr_array,
        path="",
        names=names,
        boxes=boxes,
        keypoints=keypoints,
        masks=masks,
    )


def visualize_yolo(
    img: np.ndarray,
    signal: YoloSignal,
    scale: float = 1.0,
    line_width: int = 1,
    font_size: int = 20,
    kpt_radius: int = 3,
) -> Image.Image:
    """
    Visualize signals detected by YOLO.

    Args:
        image (ndarray): The image to visualize as a NumPy array.
        signal: The signal detected by YOLO. Possible signals are YoloBBox, YoloBBoxes,
                YoloPose, YoloPoses, YoloSegment, and YoloSegments.
        scale (float): The scale factor for the image. Default is 1.0.
        line_width (int): The line width for drawing boxes and lines. Default is 1.
        font_size (int): The font size for text. Default is 20.
        kpt_radius (int): The radius for drawing keypoints. Default is 3.

    Returns:
        PIL.Image.Image: The image with the detected signals visualized.
    """
    results = _signal_to_results(img, signal)

    im_bgr = results.plot(
        line_width=line_width,
        font_size=font_size,
        kpt_radius=kpt_radius,
    )

    im_rgb = Image.fromarray(im_bgr[..., ::-1])

    if scale != 1.0:
        orig_height, orig_width = results.orig_shape
        new_size = (int(orig_width * scale), int(orig_height * scale))
        im_rgb = im_rgb.resize(new_size, Image.Resampling.LANCZOS)

    return im_rgb
