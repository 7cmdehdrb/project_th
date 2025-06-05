# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *

# TF
from tf2_ros import *

# Python
import numpy as np
import cv2
from cv_bridge import CvBridge


class FakeCameraPublisher(Node):
    def __init__(self):
        super().__init__("fake_camera_publisher")

        self.image_publisher = self.create_publisher(
            Image,
            "/camera/camera1/color/image_raw",
            qos_profile=qos_profile_system_default,
        )

        self.camera_info_publisher = self.create_publisher(
            CameraInfo,
            "/camera/camera1/color/camera_info",
            qos_profile=qos_profile_system_default,
        )

        self.bridge = CvBridge()
        self.image = cv2.imread(
            "/home/min/7cmdehdrb/ros2_ws/image_0017.png",
            cv2.IMREAD_COLOR,
        )

        hz = 30
        self.timer = self.create_timer(float(1.0 / hz), self.run)

    def run(self):
        header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id="camera1_color_optical_frame",
        )

        camera_info_msg = CameraInfo(
            header=header,
            height=720,
            width=1280,
            distortion_model="plumb_bob",
            d=[
                -0.05565285682678223,
                0.06440094113349915,
                0.00033140828600153327,
                0.0007189341704361141,
                -0.020926937460899353,
            ],
            k=[
                645.3372802734375,
                0.0,
                642.25732421875,
                0.0,
                644.5441284179688,
                380.40771484375,
                0.0,
                0.0,
                1.0,
            ],
            r=[
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            p=[
                645.3372802734375,
                0.0,
                642.25732421875,
                0.0,
                0.0,
                644.5441284179688,
                380.40771484375,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            binning_x=0,
            binning_y=0,
            roi=RegionOfInterest(
                x_offset=0,
                y_offset=0,
                height=0,
                width=0,
                do_rectify=False,
            ),
        )

        self.camera_info_publisher.publish(camera_info_msg)

        image_msg = self.bridge.cv2_to_imgmsg(self.image, "bgr8", header=header)
        self.image_publisher.publish(image_msg)


def main():
    rclpy.init(args=None)

    node = FakeCameraPublisher()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
