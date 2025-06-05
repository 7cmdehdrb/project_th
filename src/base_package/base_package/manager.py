# ROS2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Messages
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import *
from tf2_geometry_msgs.tf2_geometry_msgs import PoseStamped as TF2PoseStamped
from builtin_interfaces.msg import Duration as BuiltinDuration

# TF
from tf2_ros import *

# Python Libraries
import cv2
import numpy as np
import cv_bridge
from PIL import Image as PILImage
from PIL import ImageEnhance


class Manager(object):
    def __init__(self, node: Node, *args, **kwargs):
        self._node = node


class ImageManager(Manager):
    def __init__(
        self,
        node: Node,
        subscribed_topics: list = [],
        published_topics: list = [],
        *args,
        **kwargs,
    ):
        """
        subscribed_topics: list
            [
                {
                    "topic_name": str,
                    "callback": callable
                }
            ]
        published_topics: list
            [
                {
                    "topic_name": str
                }
            ]
        """
        super().__init__(node, *args, **kwargs)

        self._bridge = cv_bridge.CvBridge()

        self._subscribers = [
            {
                "topic_name": sub["topic_name"],
                "subscriber": self._node.create_subscription(
                    Image,
                    sub["topic_name"],
                    sub["callback"],
                    qos_profile=qos_profile_system_default,
                ),
            }
            for sub in subscribed_topics
        ]
        self._publishers = [
            {
                "topic_name": pub["topic_name"],
                "publisher": self._node.create_publisher(
                    Image,
                    pub["topic_name"],
                    qos_profile=qos_profile_system_default,
                ),
            }
            for pub in published_topics
        ]

    def get_publisher(self, topic_name: str) -> Publisher:
        for pub in self._publishers:
            if pub["topic_name"] == topic_name:
                return pub["publisher"]

    def get_subscriber(self, topic_name: str):
        for sub in self._subscribers:
            if sub["topic_name"] == topic_name:
                return sub["subscriber"]

    def encode_message(self, image: np.ndarray, encoding: str = "bgr8"):
        return self._bridge.cv2_to_imgmsg(image, encoding=encoding)

    def decode_message(self, image_msg: Image, desired_encoding: str = "bgr8"):
        return self._bridge.imgmsg_to_cv2(image_msg, desired_encoding=desired_encoding)

    def publish(self, topic_name: str, msg: Image):
        self.get_publisher(topic_name).publish(msg)
