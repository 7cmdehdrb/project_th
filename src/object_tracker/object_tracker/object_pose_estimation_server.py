# Python Standard Libraries
import io
import json
import os
import socket
import struct
import sys
import time
import argparse
import array

# Third-Party Libraries
import cv2
import numpy as np
import tqdm
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

# ROS2 Libraries
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Message Types
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import (
    BoundingBox,
    BoundingBox3D,
    BoundingBox3DMultiArray,
    BoundingBoxMultiArray,
)
from custom_msgs.srv import MegaposeRequest

# ROS2 TF
from tf2_ros import *

# Custom Modules
from base_package.header import QuaternionAngle, Queue
from base_package.manager import ImageManager, Manager, ObjectManager
from object_tracker.megapose_client import MegaPoseClient
from object_tracker.segmentation_manager import SegmentationManager
from ament_index_python.packages import get_package_share_directory


class ObjectPoseEstimationManager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node=node, *args, **kwargs)

        # >>> Client >>>
        self._srv = self._node.create_client(MegaposeRequest, "/megapose_request")

        while not self._srv.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info(
                "/megapose_request service not available, waiting again..."
            )

    def send_request(self):
        # >>> Request >>>
        request = MegaposeRequest.Request()

        # >>> Response >>>
        response: MegaposeRequest.Response = self._srv.call(request)

        return response.response


class ObjectPoseEstimator(Node):
    def __init__(self, *args, **kwargs):
        super().__init__("object_pose_estimator")

        # >>> Parameters >>>
        self._use_depth = kwargs.get("use_depth", False)
        self._refiner_iterations = kwargs.get("refiner_iterations", 5)
        # <<< Parameters <<<

        # >>> Instance Variables >>>
        subscribed_topics = [
            {
                "topic_name": "/camera/camera1/color/image_raw",
                "callback": self.image_callback,
            },
        ]

        if self._use_depth:
            subscribed_topics.append(
                {
                    "topic_name": "/camera/camera1/depth/image_rect_raw",
                    "callback": self.depth_image_callback,
                }
            )

        # TODO: FUCK

        self._megapose_client = MegaPoseClient(node=self, *args, **kwargs)
        kwargs["available_objects"] = self._megapose_client._avilable_objects

        self._object_manager = ObjectManager(node=self, *args, **kwargs)
        self._segmentation_manager = SegmentationManager(node=self, *args, **kwargs)
        self._image_manager = ImageManager(
            node=self,
            subscribed_topics=subscribed_topics,
            published_topics=[],
            *args,
            **kwargs,
        )
        # <<< Instance Variables <<<

        # >>> Data >>>
        self._image: Image = None
        self._depth_image: Image = None
        # <<< Data <<<

        # >>> Load Files >>>
        fcn_package_path = get_package_share_directory("object_tracker")

        resource_path = os.path.join(
            fcn_package_path, "../ament_index/resource_index/packages"
        )

        model_path = kwargs["obj_bounds_file"]
        if not os.path.isfile(model_path):
            model_path = os.path.join(resource_path, model_path)

        with open(model_path, "r") as f:
            self._obj_bounds = json.load(f)
        # <<< Load Files <<<

        # >>> ROS2 >>>
        self.megapose_srv = self.create_service(
            MegaposeRequest,
            "/megapose_request",
            self.megapose_request_callback,
            qos_profile=qos_profile_system_default,
        )
        # <<< ROS2 <<<

        # NO MAIN LOOP. This node is only runnning for megapose_request callbacks.

    def image_callback(self, msg: Image):
        self._image = msg
        # print("Image received.")

    def depth_image_callback(self, msg: Image):
        self._depth_image = msg
        # print("DImage received.")

    def get_object_angle(self, cTo: list) -> float:
        rotation_matrix = np.array(cTo).reshape(4, 4)[:3, :3]
        rot = R.from_matrix(rotation_matrix)

        z_world = np.array([0, 0, 1])
        y_axis = rot.apply([0, 1, 0])

        cosine_angle = np.clip(np.dot(y_axis, z_world), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cosine_angle))
        abs_angle_deg = np.abs(np.abs(angle_deg) - 90.0)

        return abs_angle_deg

    def first_megapose_request_callback(self, label: list, bbox: list) -> dict:
        # >>> STEP 1. Crop image
        if self._image is None or (self._use_depth and self._depth_image is None):
            self.get_logger().warn("No image received.")
            return None

        offset = np.random.randint(0, 10)
        offset_bbox = [
            [
                int(
                    np.clip(
                        bbox[0][0] - offset * 2,
                        0,
                        640,
                    )
                ),
                int(
                    np.clip(
                        bbox[0][1] - offset * 2,
                        0,
                        480,
                    )
                ),
                int(
                    np.clip(
                        bbox[0][2] + offset * 2,
                        0,
                        640,
                    )
                ),
                int(
                    np.clip(
                        bbox[0][3] + offset * 2,
                        0,
                        480,
                    )
                ),
            ]
        ]

        np_image = self._image_manager.decode_message(
            self._image, desired_encoding="bgr8"
        )
        np_image = self._image_manager.crop_image(np_image)

        # >>> STEP 1-2. Delete unnecessary area
        if self._use_depth:
            np_depth_image = self._image_manager.decode_message(
                self._depth_image, desired_encoding="passthrough"
            )
            np_depth_image = self._image_manager.crop_image(np_depth_image)

            zero_depth_image = (
                np.ones_like(np_depth_image, dtype=np.uint16) * 5000
            )  # 5m by default
            zero_depth_image[
                int(bbox[0][0]) : int(bbox[0][2]), int(bbox[0][1]) : int(bbox[0][3])
            ] = np_depth_image[
                int(bbox[0][0]) : int(bbox[0][2]), int(bbox[0][1]) : int(bbox[0][3])
            ]

        data = {
            "detections": offset_bbox,
            "labels": label,
            "use_depth": bool(self._use_depth),
            "refiner_iterations": int(self._refiner_iterations),
            "depth_scale_to_m": 0.001,
        }

        # >>> STEP 2. Send request
        if self._use_depth:
            results = self._megapose_client.send_pose_request_rgbd(
                image=np_image, depth=zero_depth_image, json_data=data
            )
        else:
            results = self._megapose_client.send_pose_request(
                image=np_image, json_data=data
            )

        # >>> STEP 3. Check result
        result = {
            "score": results[0]["score"],
            "cTo": results[0]["cTo"],
            "bbox": results[0]["boundingBox"],
        }
        return result

    def second_megapose_request_callback(
        self, label: list, bbox: list, cTo: list
    ) -> dict:
        # >>> STEP 1. Crop image
        if self._image is None or (self._use_depth and self._depth_image is None):
            self.get_logger().warn("No image received.")
            return

        np_image = self._image_manager.decode_message(
            self._image, desired_encoding="bgr8"
        )
        np_image = self._image_manager.crop_image(np_image)

        # >>> STEP 1-2. Delete unnecessary area
        if self._use_depth:
            np_depth_image = self._image_manager.decode_message(
                self._depth_image, desired_encoding="passthrough"
            )
            np_depth_image = self._image_manager.crop_image(np_depth_image)

            zero_depth_image = (
                np.ones_like(np_depth_image, dtype=np.uint16) * 5000
            )  # 5m by default
            zero_depth_image[
                int(bbox[0][0]) : int(bbox[0][2]), int(bbox[0][1]) : int(bbox[0][3])
            ] = np_depth_image[
                int(bbox[0][0]) : int(bbox[0][2]), int(bbox[0][1]) : int(bbox[0][3])
            ]

        data = {
            "detections": [
                [int(bbox[0][1]), int(bbox[0][0]), int(bbox[0][3]), int(bbox[0][2])]
            ],
            "labels": label,
            "initial_cTos": [cTo],
            "use_depth": bool(self._use_depth),
            "refiner_iterations": int(self._refiner_iterations),
            "depth_scale_to_m": 0.001,
        }

        # >>> STEP 2. Send request
        if self._use_depth:
            results = self._megapose_client.send_pose_request_rgbd(
                image=np_image, depth=zero_depth_image, json_data=data
            )
        else:
            results = self._megapose_client.send_pose_request(
                image=np_image, json_data=data
            )

        # >>> STEP 3. Check result
        result = {
            "score": results[0]["score"],
            "cTo": results[0]["cTo"],
            "bbox": results[0]["boundingBox"],
        }
        return result

    def megapose_request_callback(
        self, request: MegaposeRequest.Request, response: MegaposeRequest.Response
    ):
        response_msg = BoundingBox3DMultiArray()

        segmentation_datas = self._segmentation_manager.segmentation_data

        for segmentation_data in segmentation_datas:
            segmentation_data: dict

            final_result: dict = None

            label: list = segmentation_data["label"]
            bbox: array.array = segmentation_data["bbox"]
            conf = segmentation_data["conf"]

            with tqdm.tqdm(total=10) as pbar:
                # >>> STEP 1. Use segmentation data, get first cTo and score
                for fattempt in range(10):
                    first_result = self.first_megapose_request_callback(
                        label=[self._object_manager.names[label[0]]],
                        bbox=bbox,
                    )

                    if first_result is None:
                        continue

                    fscore, fcTo, fbbox = (
                        first_result["score"],
                        first_result["cTo"],
                        first_result["bbox"],
                    )
                    object_angle = self.get_object_angle(fcTo)

                    pbar.update(1)
                    pbar.set_description(
                        f"{label}({fattempt}): {fscore:.3f}/{object_angle:.2f}"
                    )

                    # >>> STEP 2. Check score

                    # >>> STEP 2-1. Case 1 - Score is high enough
                    if fscore > 0.99 and object_angle < 5.0:
                        final_result = first_result
                        break

                    # >>> STEP 2-2. Case 2 - Score is high, but validation is needed
                    elif fscore > 0.95:
                        queue = Queue(max_size=10)

                        scTo = fcTo
                        is_data_valid = False

                        # >>> STEP 2-2-1. Validation loop
                        with tqdm.tqdm(total=100) as pbar2:
                            for sattempt in range(100):

                                second_result = self.second_megapose_request_callback(
                                    label=[self._object_manager.names[label[0]]],
                                    bbox=bbox,
                                    cTo=scTo,
                                )

                                if second_result is None:
                                    continue

                                scTo, sscore, sbbox = (
                                    second_result["cTo"],
                                    second_result["score"],
                                    second_result["bbox"],
                                )

                                object_angle = self.get_object_angle(scTo)
                                avg = queue.push_and_get_average(
                                    sscore if object_angle < 10.0 else 0.0
                                )

                                pbar2.update(1)
                                pbar2.set_description(
                                    f"{label}({fattempt}-{sattempt}): {avg:.3f}/{object_angle:.2f}"
                                )

                                if avg > 0.95 and object_angle < 5.0:
                                    final_result = second_result
                                    is_data_valid = True
                                    break
                                if sattempt > 10 and avg < 0.1:
                                    break

                        # >>> STEP 2-2-2. In step 2-2-1, if data is valid, use it
                        if is_data_valid:
                            break

                    # >>> STEP 2-3. Case 3 - Score is low, try again
                    else:
                        pass

            # >>> STEP 3. Append final result to response
            if final_result is None:
                self.get_logger().warn(f"No valid result for {label}.")
                continue

            bbox_3d = self.post_process_result(result=final_result, label=label)
            response_msg.data.append(bbox_3d)

        self.get_logger().info(f"Return {len(response_msg.data)} results.")

        response.response = response_msg

        return response

    def post_process_result(self, result: dict, label: list) -> BoundingBox3D:
        # Raw result
        score = result["score"]
        cTo = result["cTo"]

        cTo_matrix = np.array(cTo).reshape(4, 4)
        # cls = self._object_manager.classes[label[0]]  # e.g. 'alive' -> 'bottle_1'
        cls = label[0]
        cls_name = self._object_manager.names[cls]

        offset_matrix = np.zeros((4, 4))
        offset_matrix[0, 3] = 0.06  # TODO: Change this value
        cTo_matrix += offset_matrix

        cTo_matrix_ros = QuaternionAngle.transform_realsense_to_ros(cTo_matrix)
        translation_matrix = cTo_matrix_ros[:3, 3] + np.array(
            [0, 0, self._obj_bounds[cls_name]["y"] / 2.0]
        )

        rotation_matrix = cTo_matrix_ros[:3, :3]
        quaternion_matrix = QuaternionAngle.quaternion_from_rotation_matrix(
            rotation_matrix
        )

        bbox_3d = BoundingBox3D(
            id=self._object_manager.indexs[cls],
            cls=cls,  # e.g. 'bottle_1'
            conf=float(score),
            pose=Pose(
                position=Point(**dict(zip(["x", "y", "z"], translation_matrix))),
                orientation=Quaternion(
                    **dict(zip(["x", "y", "z", "w"], quaternion_matrix))
                ),
            ),
            scale=Vector3(
                x=np.clip(self._obj_bounds[cls_name]["x"], 0.0, 0.05),
                y=np.clip(self._obj_bounds[cls_name]["y"], 0.0, 0.2),
                z=np.clip(self._obj_bounds[cls_name]["z"], 0.0, 0.05),
            ),
        )

        return bbox_3d


def main():
    rclpy.init(args=None)

    parser = argparse.ArgumentParser(description="FCN Server Node")

    parser.add_argument(
        "--obj_bounds_file",
        type=str,
        required=False,
        default="obj_bounds.json",
        help="Path or file name of object bounds. If input is a file name, the file should be located in the 'resource' directory. Required",
    )
    parser.add_argument(
        "--use_depth",
        type=bool,
        required=False,
        default=True,
        help="Use depth image. Default is False.",
    )

    args = parser.parse_args()
    kagrs = vars(args)

    node = ObjectPoseEstimator(**kagrs)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
