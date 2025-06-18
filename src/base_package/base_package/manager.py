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


class ObjectManager(Manager):
    def __init__(self, node: Node, *arg, **kwargs):
        super().__init__(node, *arg, **kwargs)

        self._segmentation_objects = {
            "can_1_cola": 0,
            "can_2_sikhye": 1,
            "can_3_peach": 2,
            "can_4_catata": 3,
            "cup_1_sky": 4,
            "cup_2_white": 5,
            "cup_3_blue": 6,
            "cup_4_purple": 7,
            "cup_5_green": 8,
            "mug_1_black": 9,
            "mug_2_gray": 10,
            "mug_3_yellow": 11,
            "mug_4_orange": 12,
            "bottle_1_alive": 13,
            "bottle_2_greentea": 14,
            "bottle_3_yellow": 15,
            "bottle_4_red": 16,
            "soda": 17,
        }

        self._segmentation_objects_inv = {
            v: k for k, v in self._segmentation_objects.items()
        }

        self._objects = {
            "coca_cola": 0,
            "sikhye": 1,
            "yello_peach": 2,
            "catata": 3,
            "cup_sky": 4,
            "cup_white": 5,
            "cup_blue": 6,
            "cup_purple": 7,
            "cup_green": 8,
            "mug_black": 9,
            "mug_gray": 10,
            "mug_yellow": 11,
            "mug_orange": 12,
            "alive": 13,
            "green_tea": 14,
            "yellow_smoothie": 15,
            "tomato": 16,
            "cyder": 17,
        }

        self._objects_inv = {v: k for k, v in self._objects.items()}

    def get_semtentation_label(self, label: str) -> str:
        """
        Returns the segmentation label for the given label.
        If the label is not found, returns the original label.
        """
        id = self._objects[label]
        label = self._segmentation_objects_inv[id]

        return label

    def get_object_label(self, label: str) -> str:
        """
        Returns the object label for the given label.
        If the label is not found, returns the original label.
        """
        id = self._segmentation_objects[label]
        label = self._objects_inv[id]

        return label

    @property
    def segmentation_objects(self):
        return self._segmentation_objects

    @property
    def segmentation_objects_inv(self):
        return self._segmentation_objects_inv

    @property
    def objects(self):
        return self._objects

    @property
    def objects_inv(self):
        return self._objects_inv

    @property
    def megapose_objects(self):
        return self._objects

    @property
    def megapose_objects_inv(self):
        return self._objects_inv


class TransformManager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        self._tf_buffer = Buffer(node=self._node, cache_time=Duration(seconds=2))
        self._tf_listener = TransformListener(node=self._node, buffer=self._tf_buffer)
        self._tf_broadcaster = TransformBroadcaster(self._node)

    def check_transform_valid(self, target_frame: str, source_frame: str):
        try:
            valid = self._tf_buffer.can_transform(
                target_frame,
                source_frame,
                self._node.get_clock().now().to_msg(),
                timeout=Duration(seconds=0.1),
            )

            if not valid:
                raise Exception("Transform is not valid")

            return valid
        except Exception as e:
            self._node.get_logger().warn(
                f"Cannot Lookup Transform Between {target_frame} and {source_frame}"
            )
            # self._node.get_logger().warn(e)
            return False

    def transform_pose(
        self,
        pose: Union[Pose, PoseStamped],
        target_frame: str,
        source_frame: str,
    ) -> PoseStamped:
        """
        Transform a pose from the source frame to the target frame.
        """
        if not isinstance(pose, (Pose, PoseStamped)):
            self._node.get_logger().warn("Input must be of type Pose or PoseStamped.")
            return None

        if self.check_transform_valid(target_frame, source_frame):
            try:
                transformed_pose_stamped = PoseStamped()

                if isinstance(pose, Pose):
                    pose: Pose
                    pose_stamped = TF2PoseStamped(
                        header=Header(
                            stamp=self._node.get_clock().now().to_msg(),
                            frame_id=source_frame,
                        ),
                        pose=pose,
                    )
                elif isinstance(pose, PoseStamped):
                    pose: PoseStamped
                    pose_stamped = TF2PoseStamped(
                        header=Header(
                            stamp=self._node.get_clock().now().to_msg(),
                            frame_id=source_frame,
                        ),
                        pose=pose.pose,
                    )
                else:
                    raise TypeError("Input must be of type Pose or PoseStamped.")

                transformed_data = self._tf_buffer.transform(
                    object_stamped=pose_stamped,
                    target_frame=target_frame,
                    timeout=Duration(seconds=1),
                )

                transformed_pose_stamped.header = transformed_data.header
                transformed_pose_stamped.pose = transformed_data.pose

                return transformed_pose_stamped

            except Exception as e:
                self._node.get_logger().warn(
                    f"Cannot Transform Pose from {source_frame} to {target_frame}"
                )
                # self._node.get_logger().warn(e)
                return None

        return None


class SimpleSubscriberManager(Manager):
    def __init__(self, node: Node, topic_name: str, msg_type, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        self._subscriber = self._node.create_subscription(
            msg_type,
            topic_name,
            self._callback,
            qos_profile=qos_profile_system_default,
        )

        self._data = None

    def _callback(self, msg):
        """
        Callback for receiving messages.
        Updates the data with the received message.
        """
        self._data = msg

    @property
    def subscriber(self):
        return self._subscriber

    @property
    def data(self):
        """
        Returns the last received message.
        If no message has been received, returns None.
        """
        return self._data
