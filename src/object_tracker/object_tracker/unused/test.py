# ROS2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import *
from std_msgs.msg import *
from sensor_msgs.msg import *
from rclpy.qos import qos_profile_system_default
from rclpy.time import Time

# OpenCV
import cv2
from cv_bridge import CvBridge
import rotutils

# Python
import numpy as np
from PIL import ImageEnhance
from PIL import Image as PILImage

# Megapose Server
import socket
import struct
import json
import io
import time
from scipy.spatial.transform import Rotation as R


class MegaPoseClient:
    class ServerMessage:
        GET_POSE = "GETP"
        RET_POSE = "RETP"
        ERR = "RERR"
        OK = "OKOK"

    def __init__(self, socket: socket.socket):
        self.is_segmentation_valid = False
        self.SERVER_OPERATION_CODE_LENGTH = 4

        self.result_img = None
        self.result_pose = None

        self.initial_data = {
            "detections": [[260.0, 110.0, 350.0, 343.0]],
            "labels": ["smoothie"],
            "use_depth": False,
        }

        self.loop_data = {
            "initial_cTos": None,
            "labels": ["smoothie"],
            "refiner_iterations": 1,
            "use_depth": False,
        }

        self.socket = socket

    @staticmethod
    def transform_realsense_to_ros(transform_matrix: np.array) -> np.array:
        """
        Realsense 좌표계를 ROS 좌표계로 변환합니다.

        Realsense 좌표계:
            X축: 이미지의 가로 방향 (오른쪽으로 증가).
            Y축: 이미지의 세로 방향 (아래쪽으로 증가).
            Z축: 카메라 렌즈가 바라보는 방향 (깊이 방향).

        ROS 좌표계:
            X축: 앞으로 나아가는 방향.
            Y축: 왼쪽으로 이동하는 방향.
            Z축: 위로 이동하는 방향.

        Args:
            transform_matrix (np.ndarray): 4x4 변환 행렬.

        Returns:
            np.ndarray: 변환된 4x4 변환 행렬 (ROS 좌표계 기준).
        """
        if transform_matrix.shape != (4, 4):
            raise ValueError("Input transformation matrix must be a 4x4 matrix.")

        # Realsense에서 ROS로 좌표계를 변환하는 회전 행렬
        realsense_to_ros_rotation = np.array(
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]  # X -> Z  # Y -> -X  # Z -> -Y
        )

        # 변환 행렬의 분해
        rotation = transform_matrix[:3, :3]  # 3x3 회전 행렬
        translation = transform_matrix[:3, 3]  # 3x1 평행 이동 벡터

        # 좌표계 변환
        rotation_ros = realsense_to_ros_rotation @ rotation
        translation_ros = realsense_to_ros_rotation @ translation

        # 새로운 변환 행렬 구성
        transform_matrix_ros = np.eye(4)
        transform_matrix_ros[:3, :3] = rotation_ros
        transform_matrix_ros[:3, 3] = translation_ros

        return transform_matrix_ros

    def loop(self, frame: np.array):
        bbox = False

        # If bbox is already available, skip the detection step
        if bbox:
            x1, y1, x2, y2 = map(int, self.initial_data["detections"][0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.result_img = frame
            self.result_pose = PoseStamped()

            return

        cto, score, bbox = self.send_pose_request(
            self.socket,
            frame,
            (self.initial_data if not self.is_segmentation_valid else self.loop_data),
        )

        if score is not None:
            if score < 0.5:
                self.is_segmentation_valid = False
                print("Score too low, skipping")

                # Overlay the bounding box on the image
                x1, y1, x2, y2 = map(int, self.initial_data["detections"][0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            else:
                self.is_segmentation_valid = True
                self.loop_data["initial_cTos"] = [cto.tolist()]

                # Overlay the bounding box on the image
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Overlay Score on the image
                cv2.putText(
                    frame,
                    f"Score: {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

                self.result_img = frame

                # Get PoseStamped message
                cto_matrix = np.array(cto).reshape(4, 4)
                transformed_cto_matrix = self.transform_realsense_to_ros(cto_matrix)
                translation = transformed_cto_matrix[:3, 3]
                rotation_matrix = transformed_cto_matrix[:3, :3]
                rotation = R.from_matrix(rotation_matrix).as_quat()

                pose_msg = PoseStamped(
                    header=Header(
                        stamp=Time().to_msg(),
                        frame_id="camera1_link",
                    ),
                    pose=Pose(
                        position=Point(
                            x=translation[0], y=translation[1], z=translation[2]
                        ),
                        orientation=Quaternion(
                            x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3]
                        ),
                    ),
                )

                self.result_pose = pose_msg

    def send_message(self, sock: socket.socket, code: str, data: bytes):
        msg_length = struct.pack(">I", len(data))
        sock.sendall(msg_length + code.encode("UTF-8") + data)

    def receive_message(self, sock: socket.socket):
        msg_length = sock.recv(4)
        length = struct.unpack(">I", msg_length)[0]
        code = sock.recv(self.SERVER_OPERATION_CODE_LENGTH).decode("UTF-8")
        data = sock.recv(length)
        return code, io.BytesIO(data)

    def pack_string(self, data: str) -> bytes:
        encoded = data.encode("utf-8")
        length = struct.pack(">I", len(encoded))
        return length + encoded

    def read_string(self, buffer: io.BytesIO) -> str:
        length = struct.unpack(">I", buffer.read(4))[0]
        return buffer.read(length).decode("utf-8")

    def send_pose_request(
        self, sock: socket.socket, image: np.ndarray, json_data: dict
    ):
        # **(1) RGB 이미지를 전송할 수 있도록 BGR → RGB 변환**
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # **(2) 서버의 read_image 형식에 맞춰 (height, width, channels) 전송**
        height, width, channels = image.shape
        img_shape_bytes = struct.pack(">3I", height, width, channels)
        img_bytes = image.tobytes()

        # **(3) JSON 데이터를 직렬화**
        json_str = json.dumps(json_data)
        json_bytes = self.pack_string(json_str)

        # **(4) 최종 데이터 생성 (크기 + 이미지 + JSON)**
        data = img_shape_bytes + img_bytes + json_bytes

        # **(5) 서버에 데이터 전송**
        self.send_message(sock, MegaPoseClient.ServerMessage.GET_POSE, data)

        # **(6) 서버 응답 수신**
        code, response_buffer = self.receive_message(sock)
        if code == MegaPoseClient.ServerMessage.RET_POSE:
            json_str = self.read_string(response_buffer)
            decoded_json = json.loads(json_str)

            if len(decoded_json) == 1:
                data = decoded_json[0]

                cto = np.array(data["cTo"])
                score = data["score"]
                bbox = data["boundingBox"]

                return cto, score, bbox

        elif code == MegaPoseClient.ServerMessage.ERR:
            print("Error from server:", self.read_string(response_buffer))
        else:
            print("Unknown response code:", code)

        return None, None, None


class CameraViewer(Node):
    def __init__(self):
        super().__init__("camera_viewer_node")

        self.SERVER_HOST = "127.0.0.1"
        self.SERVER_PORT = 5555

        self.bridge = CvBridge()

        # ROS
        self.publisher = self.create_publisher(
            Image,
            "/segmented_image",
            qos_profile=qos_profile_system_default,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            "/megapose",
            qos_profile=qos_profile_system_default,
        )
        self.subscriber = self.create_subscription(
            Image,
            "/camera/camera1/color/image_raw",
            self.image_callback,
            qos_profile=qos_profile_system_default,
        )
        self.image = None

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.socket.connect((self.SERVER_HOST, self.SERVER_PORT))
        print(f"Connected to {self.SERVER_HOST}:{self.SERVER_PORT}")

        self.megapose_client = MegaPoseClient(socket=self.socket)

        self.create_timer(0.1, self.run)
        # self.run()

    def run(self):
        # while True:
        if self.image is None:
            self.get_logger().info("No image received yet.")
            return

        frame = self.bridge.imgmsg_to_cv2(self.image, desired_encoding="bgr8")

        def crop_and_resize_image(img: np.asarray) -> np.asarray:
            desired_width, desired_height = 640, 480

            target_width = img.shape[1] * (desired_height / desired_width)

            # Crop image
            cropped_img = img[
                :,
                int((img.shape[1] - target_width) // 2) : int(
                    (img.shape[1] + target_width) // 2
                ),
                :,
            ]

            # Resize image
            resized_img = cv2.resize(cropped_img, (desired_width, desired_height))

            return resized_img

        # resized_frame = crop_and_resize_image(frame)

        self.megapose_client.loop(frame=frame)

        result_img = self.megapose_client.result_img
        result_pose = self.megapose_client.result_pose

        if result_img is not None and result_pose is not None:
            self.publisher.publish(
                self.bridge.cv2_to_imgmsg(result_img, encoding="bgr8")
            )
            self.pose_publisher.publish(result_pose)

    def image_callback(self, msg: Image):
        self.image = msg

        # self.image_loop()

    def image_loop(self):
        if self.image is None:
            return None

        msg = self.image

        # Load Image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv_image_array = np.asanyarray(cv_image)

        assert cv_image_array.shape[0] == 720 and cv_image_array.shape[1] == 1280

        def crop_and_resize_image(img: np.asarray) -> np.asarray:
            desired_width, desired_height = 640, 480

            target_width = img.shape[1] * (desired_height / desired_width)

            # Crop image
            cropped_img = img[
                :,
                int((img.shape[1] - target_width) // 2) : int(
                    (img.shape[1] + target_width) // 2
                ),
                :,
            ]

            # Resize image
            resized_img = cv2.resize(cropped_img, (desired_width, desired_height))

            return resized_img

        transformed_img = crop_and_resize_image(cv_image_array)

        # 결과 화면 출력
        cv2.imshow("YOLOv11 Segmentation (Cropped 640x480)", transformed_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = CameraViewer()

    rclpy.spin(node=node)

    cv2.destroyAllWindows()

    node.destroy_node()


if __name__ == "__main__":
    main()
