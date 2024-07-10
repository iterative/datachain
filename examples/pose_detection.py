import json
import os

import fsspec
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToDict, ParseDict
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageFile
from tabulate import tabulate

from datachain.catalog import get_catalog
from datachain.query import C, DatasetQuery, DatasetRow, Stream, udf
from datachain.sql.types import JSON


def load_image(stream):
    with stream:
        img = Image.open(stream)
        img.load()
    format = img.format  # typically, this will be JPEG
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))
    return image, format


def landmarks_list_to_pb2(pose_landmarks):
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend(
        [
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ]
    )
    return pose_landmarks_proto


def pb2_to_dict(pose_landmarks_proto):
    pose_dict = MessageToDict(pose_landmarks_proto)
    # Test round trip while we are at it
    assert pose_landmarks_proto == ParseDict(
        pose_dict, landmark_pb2.NormalizedLandmarkList()
    )
    return pose_dict


def annotate_image(image, pose):
    annotated_image = np.copy(image)
    for pose_landmarks_proto in pose:
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def mask_image(detection_result):
    # This only takes the most confident detection.
    # We could union all the detections or do a generator over each.
    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    return visualized_mask.astype(np.uint8)


@udf(
    params=(Stream(), *tuple(DatasetRow.schema.keys())),
    output={**DatasetRow.schema, "pose": JSON},
)
class PoseDetector:
    annotated_folder = "annotated_images"
    mask_folder = "mask_images"

    def __init__(
        self,
        *,
        model_asset_path,
        bucket_name,
        prefix,
    ):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options, output_segmentation_masks=True
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        catalog = get_catalog()
        self.client, _ = catalog.parse_url(os.path.join(prefix, bucket_name))

    def save(self, image, source, folder, name, format):
        # Do writeback
        blob_name = os.path.join(folder, name)
        urlpath = os.path.join(source, blob_name)
        cloud_file = fsspec.open(urlpath=urlpath, mode="wb")
        with cloud_file as fp:
            image.save(fp, format=format)

        # Get the blob info
        info_ = self.client.fs.info(urlpath)
        entry = self.client.convert_info(info_, folder)
        return DatasetRow.create(
            name=name,
            source=source,
            parent=folder,
            size=entry.size,
            location=None,
            vtype="",
            dir_type=0,
            owner_name=entry.owner_name,
            owner_id=entry.owner_id,
            is_latest=entry.is_latest,
            last_modified=entry.last_modified,
            version=entry.version,
            etag=entry.etag,
        )

    def __call__(
        self,
        stream,
        *args,
    ):
        # Build a dict from row contents
        record = dict(zip(DatasetRow.schema.keys(), args))

        # Don't re-apply analysis to output
        if record["parent"] in (self.annotated_folder, self.mask_folder):
            return

        # CLeanup to records
        del record["random"]  # random will be populated automatically
        record["is_latest"] = record["is_latest"] > 0  # needs to be a bool
        row = DatasetRow.create(**record)

        # Put into media pipe object
        image, image_format = load_image(stream)

        # Do the detection
        detection_result = self.detector.detect(image)

        # Move into protobuf list for next step
        pose = [landmarks_list_to_pb2(lm) for lm in detection_result.pose_landmarks]

        # Turn into json
        pose_json = [pb2_to_dict(lmp) for lmp in pose]

        # Yield same row back (with json pose info)
        yield (*row, json.dumps(pose_json))

        # No detections ==> we can stop here
        if len(pose) == 0:
            return

        # Annotate image
        annotated_image = annotate_image(image.numpy_view(), pose)
        annotated_image = Image.fromarray(annotated_image)  # make PIL object

        # Save the image and get the cloud object info
        row = self.save(
            image=annotated_image,
            source=record["source"],
            folder=self.annotated_folder,
            name=record["name"],
            format=image_format,
        )
        yield (*row, json.dumps(pose_json))

        # Now do mask
        visualized_mask = mask_image(detection_result)
        visualized_mask = Image.fromarray(visualized_mask)  # make PIL object

        # Save the image and get the cloud object info
        row = self.save(
            image=visualized_mask,
            source=record["source"],
            folder=self.mask_folder,
            name=record["name"],
            format=image_format,
        )
        yield (*row, json.dumps(pose_json))


cloud_prefix = "s3://"  # for GCP just switch to "gs://"
bucket = "dvcx-50k-laion-files-writable"  # which bucket to use for both read and write
bucket_region = "us-east-2"  # no need to specify for GCP

# Use: !wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
model_asset_path = "pose_landmarker.task"

file_type = "*.jpg"  # which files to use
filter_mod = 512  # how much of a subset of the data to use, i.e., 1/512
chunk_num = 1

# only needed for AWS (no effect if using GCP)
os.environ["AWS_REGION"] = bucket_region
os.environ["AWS_DEFAULT_REGION"] = bucket_region

ImageFile.LOAD_TRUNCATED_IMAGES = True

assert chunk_num < filter_mod

pose_udf = PoseDetector(
    model_asset_path=model_asset_path,
    bucket_name=bucket,
    prefix=cloud_prefix,
)

if __name__ == "__main__":
    data = (
        DatasetQuery(os.path.join(cloud_prefix, bucket))
        .filter(C.name.glob(file_type))
        .filter(C.random % filter_mod == chunk_num)
        .generate(pose_udf)
        .results()
    )

    # Output the contents of the new dataset.
    print(tabulate(data))
