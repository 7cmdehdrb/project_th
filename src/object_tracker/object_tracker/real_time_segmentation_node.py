# Python
import argparse
import json
import os
import sys
from PIL import Image as PILImage
from PIL import ImageEnhance

# OpenCV
import cv2

# NumPy
import numpy as np

# ROS2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Messages
from custom_msgs.msg import BoundingBox, BoundingBoxMultiArray
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *

# TF
from tf2_ros import *

# YOLO
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Masks, Results

# Custom Packages
from ament_index_python.packages import get_package_share_directory
from base_package.manager import ImageManager, Manager


class YoloManager(Manager):
    def __init__(self, node: Node, *arg, **kwargs):
        super().__init__(node, *arg, **kwargs)

        # >>> Load Files >>>
        tracker_package_path = get_package_share_directory("object_tracker")

        resource_path = os.path.join(
            tracker_package_path, "../ament_index/resource_index/packages"
        )

        model_path = kwargs["model_file"]
        if not os.path.isfile(model_path):
            model_path = os.path.join(resource_path, model_path)
        # <<< Load Files <<<

        # Load YOLO v11 Model
        self._model = YOLO(kwargs["model_file"], verbose=False)
        self._model.eval()

    def predict(self, image: PILImage):
        return self._model(image, verbose=False)


class RealTimeSegmentationNode(Node):
    def __init__(
        self,
        *arg,
        **kwargs,
    ):
        super().__init__("real_time_segmentation_node")

        # >>> Managers >>>
        self._yolo_manager = YoloManager(self, *arg, **kwargs)

        self._image_manager = ImageManager(
            self,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/color/image_raw",
                    "callback": self._image_callback,
                },
            ],
            published_topics=[
                {"topic_name": self.get_name() + "/segmented_image"},
            ],
            *arg,
            **kwargs,
        )
        self._camera_image: Image = None
        # <<< Managers <<<

        # >>> ROS2 >>>
        self.segmented_bbox_publisher = self.create_publisher(
            BoundingBoxMultiArray,
            self.get_name() + "/segmented_bbox",
            qos_profile=qos_profile_system_default,
        )
        # <<< ROS2 <<<

        # >>> Parameters >>>
        self._conf_threshold = float(kwargs["conf_threshold"])
        self._color_dict: dict = None  # Dictionary to store colors for each class
        # <<< Parameters <<<

    def _image_callback(self, msg: Image):
        self._camera_image = msg

    def _publish_bbox_image(self, bboxes: BoundingBoxMultiArray) -> None:
        np_image = self._image_manager.decode_message(
            self._camera_image, desired_encoding="rgb8"
        )  # TODO: Make sure the image is in RGB format

        for bbox in bboxes.data:
            bbox: BoundingBox

            x1, y1, x2, y2 = bbox.bbox

            # 바운딩 박스 그리기
            label = f"{bbox.cls}, {bbox.conf:.2f}"
            cv2.rectangle(
                np_image,
                (x1, y1),
                (x2, y2),
                self._color_dict.get(
                    bbox.cls, (255, 0, 0)
                ),  # Default to red if class not found
                2,
            )
            cv2.putText(
                img=np_image,
                text=label,
                org=(x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=self._color_dict.get(
                    bbox.cls, (255, 0, 0)
                ),  # Default to red if class not found),  # White text for better visibility
                thickness=2,
            )

        segmented_image: Image = self._image_manager.encode_message(
            np_image, encoding="rgb8"
        )

        self._image_manager.publish(
            topic_name=self.get_name() + "/segmented_bbox",
            msg=segmented_image,
        )

        return segmented_image

    def _do_segmentation(self, msg: Image) -> BoundingBoxMultiArray | None:
        if self._camera_image is None:
            self.get_logger().warn("No camera image received yet.")
            return None

        # Load Image
        np_image = self._image_manager.decode_message(
            msg, desired_encoding="rgb8"
        )  # TODO: Make sure the image is in RGB format
        pil_image = PILImage.fromarray(np_image)

        # YOLO 세그멘테이션 수행
        result: Results = self._yolo_manager.predict(pil_image)[0]

        boxes: Boxes = result.boxes
        classes: dict = result.names

        np_boxes = boxes.xyxy.cpu().numpy()
        np_confs = boxes.conf.cpu().numpy()
        np_cls = boxes.cls.cpu().numpy()

        # Make color dictionary for bounding boxes
        if self._color_dict is None:
            color_dict = {}
            for idx, cls in enumerate(classes):
                # Generate a random color for each class
                color = tuple(np.random.randint(0, 255, size=3).tolist())
                color_dict[cls] = color

            self._color_dict = color_dict

        # 바운딩 박스 그리기
        bboxes = BoundingBoxMultiArray()

        for idx in range(len(boxes)):
            id = int(np_cls[idx])  # Unique ID for the object. (int)

            cls = str(classes[id])  # Object class name (str)
            conf = float(np_confs[idx])  # Confidence score (float)

            bbox = map(int, np_boxes[idx])  # Bounding box coordinates (int)

            # >>> STEP 1. 신뢰도 확인
            if conf < self._conf_threshold:
                continue

            # >>> STEP 2. 바운딩 박스 추가
            bboxes.data.append(BoundingBox(id=id, cls=cls, conf=conf, bbox=bbox))

        return bboxes


def main(args=None):
    rclpy.init(args=args)

    from rclpy.utilities import remove_ros_args
    from base_package.header import str2bool

    # Remove ROS2 arguments
    argv = remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser(description="FCN Server Node")
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="(Required) Path or file name of the model. If input is a file name, the file should be located in the 'resource' directory.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        required=False,
        default=0.7,
        help="(Optional)Confidence threshold for object detection. Default is 0.7",
    )

    args = parser.parse_args(argv[1:])
    kagrs = vars(args)

    node = RealTimeSegmentationNode(**kagrs)

    rclpy.spin(node=node)

    node.destroy_node()


if __name__ == "__main__":
    main()
