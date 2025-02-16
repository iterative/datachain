# YOLO (Ultralytics) models

YOLO (You Only Look Once) by Ultralytics is a state-of-the-art object detection framework
that provides fast and accurate real-time predictions. Developed by Ultralytics,
it supports tasks like object detection, instance segmentation, and pose estimation.
The framework is easy to use, highly optimized, and supports training and inference
with PyTorch.

For more details, visit [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

### Example

See more use cases in the datachain repo [examples](https://github.com/iterative/datachain/tree/main/examples).

```python
from ultralytics import YOLO

from datachain import DataChain, ImageFile
from datachain.model.yolo import YoloBox


def process_boxes(yolo: YOLO, file: ImageFile) -> YoloBox:
    results = yolo(file.read(), verbose=False)
    return YoloBox.from_yolo_results(results)


(
    DataChain.from_storage("gs://datachain-demo/coco2017/images", type="image")
    .limit(20)
    .setup(yolo=lambda: YOLO("yolo11n.pt"))
    .map(boxes=process_boxes)
    .show()
)
```

::: datachain.model.yolo.YoloBox

::: datachain.model.yolo.YoloObb

::: datachain.model.yolo.YoloSeg

::: datachain.model.yolo.YoloPose

::: datachain.model.yolo.YoloPoseBodyPart

::: datachain.model.yolo.YoloCls
