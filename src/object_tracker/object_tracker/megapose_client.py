# Python
import io
import json
import socket
import struct
import time
import os
import sys
import argparse
from enum import Enum

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

# Custom Packages
from base_package.manager import ImageManager


class MegaPoseClient(object):
    """Socket client for MegaPose server."""

    class ServerMessage:
        GET_POSE = "GETP"
        RET_POSE = "RETP"
        GET_VIZ = "GETV"
        RET_VIZ = "RETV"
        SET_INTR = "INTR"
        GET_SCORE = "GSCO"
        RET_SCORE = "RSCO"
        SET_SO3_GRID_SIZE = "SO3G"
        GET_LIST_OBJECTS = "GLSO"
        RET_LIST_OBJECTS = "RLSO"
        ERR = "RERR"
        OK = "OKOK"

    def __init__(self, node: Node, *args, **kwargs):
        """
        kwargs:
            --host: str
            --port: int
        """
        self._node = node

        # >>> Arguments >>>
        self._host = kwargs.get("host", "127.0.0.1")
        self._port = kwargs.get("port", 5555)
        # <<< Arguments <<<

        # >>> Socket >>>
        self._SERVER_OPERATION_CODE_LENGTH = 4

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._host, self._port))
        # >>> Socket >>>

        # >>> ROS2 >>>
        self._camera_info_subscriber = self._node.create_subscription(
            CameraInfo,
            "/camera/camera1/color/camera_info",
            self._camera_info_callback,
            qos_profile=qos_profile_system_default,
        )
        # <<< ROS2 <<<

        # >>> Data >>>
        self._is_configured = (
            False  # Whether the client is configured with camera intrinsics
        )
        self._image_size = (480, 640)  # Default image size (height, width)
        self._avilable_objects = self._send_list_objects_request()
        # <<< Data <<<

    @property
    def is_configured(self) -> bool:
        """
        Returns whether the client is configured with camera intrinsics.
        """
        return self._is_configured

    def _camera_info_callback(self, msg: CameraInfo):
        if self._is_configured is True:
            return None

        if (msg.height, msg.width) != self._image_size:
            self._node.get_logger().warn(
                f"Camera info size mismatch: {msg.height}x{msg.width} != {self._image_size[0]}x{self._image_size[1]}"
            )
            raise ValueError(
                f"Camera info size mismatch: {msg.height}x{msg.width} != {self._image_size[0]}x{self._image_size[1]}"
            )

        K = np.array(msg.k).reshape(3, 3)

        response = self._send_intrinsics_request(
            K=K,
            image_size=self._image_size,
        )

        if response:
            self._node.get_logger().info("Set intrinsics successfully.")
            self._is_configured = True

    def _send_intrinsics_request(self, K: np.ndarray, image_size: tuple):
        """
        서버에 카메라의 내부 파라미터(K 행렬)와 이미지 크기를 설정하는 요청을 보낸다.

        :param sock: 열린 소켓 객체
        :param K: 3x3 카메라 내부 파라미터 행렬
        :param image_size: (height, width) 이미지 크기
        """
        # K 행렬에서 필요한 파라미터 추출
        px, py = K[0, 0], K[1, 1]  # 초점 거리 (f_x, f_y)
        u0, v0 = K[0, 2], K[1, 2]  # 주점 (principal point)
        h, w = image_size  # 이미지 높이, 너비

        # JSON 데이터 생성
        intrinsics_data = {"px": px, "py": py, "u0": u0, "v0": v0, "h": h, "w": w}

        # JSON 직렬화
        json_str = json.dumps(intrinsics_data)
        json_bytes = self._pack_string(json_str)

        # 메시지 전송
        self._send_message("INTR", json_bytes)

        # 응답 수신
        code, response_buffer = self._receive_message()
        if code == MegaPoseClient.ServerMessage.OK:
            self._node.get_logger().info("Intrinsics successfully set on the server.")
            return True

        elif code == MegaPoseClient.ServerMessage.ERR:
            self._node.get_logger().warn(
                "Error from server:", self._read_string(response_buffer)
            )
        else:
            self._node.get_logger().warn("Unknown response code:", code)

        return False

    def _send_message(self, code: str, data: bytes):
        msg_length = struct.pack(">I", len(data))
        self._socket.sendall(msg_length + code.encode("UTF-8") + data)

    def _receive_message(self):
        msg_length = self._socket.recv(4)
        length = struct.unpack(">I", msg_length)[0]
        code = self._socket.recv(self._SERVER_OPERATION_CODE_LENGTH).decode("UTF-8")
        data = self._socket.recv(length)
        return code, io.BytesIO(data)

    def _pack_string(self, data: str) -> bytes:
        encoded = data.encode("utf-8")
        length = struct.pack(">I", len(encoded))
        return length + encoded

    def _read_string(self, buffer: io.BytesIO) -> str:
        length = struct.unpack(">I", buffer.read(4))[0]
        return buffer.read(length).decode("utf-8")

    def send_pose_request_rgbd(
        self, image: np.ndarray, depth: np.ndarray, json_data: dict
    ):
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # RGB image
        height, width, channels = image.shape
        img_shape_bytes = struct.pack(">3I", height, width, channels)
        img_bytes = image.tobytes()

        # JSON
        json_str = json.dumps(json_data)
        json_bytes = self._pack_string(json_str)

        # Depth image
        assert depth.dtype == np.uint16
        assert depth.shape == (height, width)

        depth_shape_bytes = struct.pack(">2I", height, width)  # only height & width
        endianness_byte = struct.pack("c", b">")  # big endian
        depth_bytes = depth.tobytes()

        # 최종 데이터 조합
        data = (
            img_shape_bytes
            + img_bytes
            + json_bytes
            + depth_shape_bytes
            + endianness_byte
            + depth_bytes
        )

        # Send and receive
        self._send_message(MegaPoseClient.ServerMessage.GET_POSE, data)
        code, response_buffer = self._receive_message()

        if code == MegaPoseClient.ServerMessage.RET_POSE:
            return json.loads(self._read_string(response_buffer))
        elif code == MegaPoseClient.ServerMessage.ERR:
            self._node.get_logger().warn(
                "Error from server:", self._read_string(response_buffer)
            )
        else:
            self._node.get_logger().error("Unknown response code:", code)
        return None

    def send_pose_request_rgb(self, image: np.ndarray, json_data: dict):
        # **(1) RGB 이미지를 전송할 수 있도록 BGR → RGB 변환**
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # **(2) 서버의 read_image 형식에 맞춰 (height, width, channels) 전송**
        height, width, channels = image.shape
        img_shape_bytes = struct.pack(">3I", height, width, channels)
        img_bytes = image.tobytes()

        # **(3) JSON 데이터를 직렬화**
        json_str = json.dumps(json_data)
        json_bytes = self._pack_string(json_str)

        # **(4) 최종 데이터 생성 (크기 + 이미지 + JSON)**
        data = img_shape_bytes + img_bytes + json_bytes

        # **(5) 서버에 데이터 전송**
        self._send_message(MegaPoseClient.ServerMessage.GET_POSE, data)

        # **(6) 서버 응답 수신**
        code, response_buffer = self._receive_message()
        if code == MegaPoseClient.ServerMessage.RET_POSE:
            json_str = self._read_string(response_buffer)
            decoded_json = json.loads(json_str)

            if len(decoded_json) > 0:
                return decoded_json

        elif code == MegaPoseClient.ServerMessage.ERR:
            print("Error from server:", self._read_string(response_buffer))
        else:
            print("Unknown response code:", code)

        return None

    def _send_list_objects_request(self):
        """
        서버에 오브젝트 목록을 요청하고 응답을 받는다.

        :param sock: 열린 소켓 객체
        :return: 오브젝트 목록 (list of str) 또는 None
        """
        # 서버에 'GLSO' 요청 전송
        self._send_message("GLSO", b"")

        # 응답 수신
        code, response_buffer = self._receive_message()

        if code == MegaPoseClient.ServerMessage.RET_LIST_OBJECTS:
            json_str = self._read_string(response_buffer)
            object_list = json.loads(json_str)
            return object_list

        elif code == MegaPoseClient.ServerMessage.ERR:
            self._node.get_logger().warn(
                f"Error from server: {self._read_string(response_buffer)}"
            )
        else:
            self._node.get_logger().warn(f"Unknown response code: {code}")

        return None


class MegaPoseClientStatus(Enum):
    NOT_CONFIGURED = 0
    SEARCHING = 1
    TRACKING = 2
    LOST = 3


class MegaPoseNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__("megapose_client_node", *args, **kwargs)

        self._target_object = kwargs.get("target_object", None)
        self._refiner_iterations = kwargs.get("refiner_iterations", 1)

        # >>> MegaPose Client Initialization >>>

        self._status = MegaPoseClientStatus.NOT_CONFIGURED
        self._methods = {
            MegaPoseClientStatus.NOT_CONFIGURED: self._not_configured,
            MegaPoseClientStatus.SEARCHING: self._searching,
            MegaPoseClientStatus.TRACKING: self._tracking,
            MegaPoseClientStatus.LOST: self._lost,
        }

        self._megapose_client = MegaPoseClient(self, *args, **kwargs)
        self._data = {
            "cTo": np.eye(4, 4),  # Initial camera-to-object transformation matrix
            "score": 0.0,  # Initial score
        }

        # >>> Bounding Box Subscriber >>>
        self._bbox_subscriber = self.create_subscription(
            BoundingBoxMultiArray,
            "",  # TODO: Set the correct topic name for bounding boxes
            self._bbox_callback,
            qos_profile=qos_profile_system_default,
        )
        self._bbox: BoundingBoxMultiArray = None

        # >>> Image Subscriber >>>
        self._image_manager = ImageManager(
            node=self,
            published_topics=[],
            subscribed_topics=[
                {
                    "topic_name": "",  # TODO: Set the correct topic name for images
                    "callback": self._image_callback,
                }
            ],
        )
        self._image: Image = None

    def _bbox_callback(self, msg: BoundingBoxMultiArray):
        self._bbox = msg

    def _image_callback(self, msg: Image):
        self._image = msg

    def _get_target_bounding_box(self) -> List[int, int, int, int]:
        """
        Get the bounding box for the target object from the received bounding boxes.
        Returns a list of [x_min, y_min, x_max, y_max] or None if not found.

        Raise:
        - None: If the target object is not found in the bounding boxes.
        """
        for bbox in self._bbox.data:
            bbox: BoundingBox

            if bbox.cls == self._target_object:
                return bbox.bbox

        return None

    def _post_process_result(self, response: dict) -> dict:
        if response is None:
            self.get_logger().warn(
                f"Failed to get initial cTos for '{self._target_object}'. Response is None."
            )
            return None

        # If the score is above a threshold, update the status to TRACKING
        if response["score"] > 0.8:
            self.get_logger().info(
                f"Initial cTos for '{self._target_object}' received successfully."
            )
            self.get_logger().info(f"Score: {response['score']:.2f}")

            result = {
                "cTo": np.array(response["cTo"]).reshape(4, 4),  # Convert to 4x4 matrix
                "score": response["score"],
            }

            # Update the status to TRACKING
            self._update_status()

            return result

        else:
            self.get_logger().warn(
                f"Failed to get initial cTos for '{self._target_object}'. Score: {result['score']:.2f}"
            )
            return None

    def run(self):
        """
        Main loop for the MegaPose client node.
        This method will be called to start the node's functionality.
        """
        # Call the current status method
        self._methods[self._status]()

    def _update_status(self) -> MegaPoseClientStatus:
        """
        Set the current status of the MegaPose client.
        This will change the behavior of the run method.
        """

        if self._status == MegaPoseClientStatus.LOST:
            self.get_logger().warn("Tracking lost, trying to reinitialize.")
            self._status = MegaPoseClientStatus.SEARCHING

        else:
            self._status = MegaPoseClientStatus(self._status.value + 1)

        return self._status

    def _not_configured(self):
        """
        Not configured state: Waiting for camera intrinsics to be set, image to be received, and bounding box to be available.
        """
        if self._target_object is None:
            self.get_logger().error("Target object is not specified.")
            exit(1)
            return None

        if not self._megapose_client._is_configured:
            self.get_logger().warn(
                "MegaPose client is not configured. Waiting for camera intrinsics."
            )
            return None

        if self._image is None:
            self.get_logger().warn("Waiting for image to be received.")
            return None

        if self._bbox is None:
            self.get_logger().warn("Waiting for bounding box to be received.")
            return None

        self._update_status()

    def _searching(self):
        """
        Searching state: Trying to get initial cTos for tracking.
        """
        # 0. Decode the image message to a NumPy array
        np_image = self._image_manager.decode_message(
            self._image,
            desired_encoding="bgr8",  # TODO: Check desired encoding is "bgr8"
        )

        # 1. Get the bounding box for the target object
        bbox = self._get_target_bounding_box()

        if bbox is None:
            self.get_logger().warn(
                f"Bounding box for '{self._target_object}' not found. Waiting for bounding box."
            )
            return None

        data = {
            "detections": [bbox],
            "labels": [self._target_object],
            "use_depth": False,
            "refiner_iterations": self._refiner_iterations,  # Number of iterations for the refiner
        }

        # 2. Send the request to the MegaPose server
        response = self._megapose_client.send_pose_request_rgb(
            image=np_image, json_data=data
        )

        # 3. Check the response
        result = self._post_process_result(response)

        if result is None:
            self.get_logger().warn(
                f"Failed to get initial cTos for '{self._target_object}'. Retrying..."
            )
            return None

        self._data = result
        self._update_status()

    def _tracking(self):
        """
        Tracking state: Actively tracking the object.
        """
        # 0. Decode the image message to a NumPy array
        np_image = self._image_manager.decode_message(
            self._image,
            desired_encoding="bgr8",  # TODO: Check desired encoding is "bgr8"
        )

        cTo: np.ndarray = self._data["cTo"]

        data = {
            "initial_cTos": cTo.flatten().tolist(),
            "labels": [self._target_object],
            "refiner_iterations": self._refiner_iterations,  # Number of iterations for the refiner
            "use_depth": False,  # Whether to use depth information
        }

        # 1. Send the request to the MegaPose server
        response = self._megapose_client.send_pose_request_rgb(
            image=np_image, json_data=data
        )

        # 2. Check the response
        result = self._post_process_result(response)

        if result is None:
            self.get_logger().warn(
                f"Tracking lost for '{self._target_object}'. Trying to reinitialize."
            )
            self._update_status()
            return None

    def _lost(self):
        """
        Lost state: Object lost, trying to reinitialize tracking.
        """
        self._update_status()


def main(args=None):
    rclpy.init(args=args)

    from rclpy.utilities import remove_ros_args
    from base_package.header import str2bool

    # Remove ROS2 arguments
    argv = remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser(description="FCN Server Node")

    parser.add_argument(
        "--host",
        type=str,
        required=False,
        default="127.0.0.1",
        help="(Optional) Host address of the MegaPose server. Default is 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=5555,
        help="(Optional) Port number of the MegaPose server. Default is 5555",
    )
    parser.add_argument(
        "--refiner_iterations",
        type=int,
        required=False,
        default=1,
        help="(Optional) Number of iterations for the refiner. Default is 1",
    )
    parser.add_argument(
        "--target_object",
        type=str,
        required=True,
        help="(Required) Name of the target object to track.",
    )

    args = parser.parse_args(argv[1:])
    kagrs = vars(args)

    node = MegaPoseNode(node=node, **kagrs)
