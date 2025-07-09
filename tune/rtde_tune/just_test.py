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
from base_package.manager import SimpleSubscriberManager


class JustTestNode(Node):
    def __init__(self):
        super().__init__("just_test_node")

        self._pub = self.create_publisher(
            Float32MultiArray, "/test_topic", QoSProfile(depth=10)
        )

        self._joint_state_subscriber = SimpleSubscriberManager(
            self, "/joint_states", JointState
        )
        self._sim_joint_state_subscriber = SimpleSubscriberManager(
            self, "/issac_sim/joint_states", JointState
        )

        self._timer = self.create_timer(0.03, self.timer_callback)

    def timer_callback(self):
        if (
            not self._joint_state_subscriber.data
            or not self._sim_joint_state_subscriber.data
        ):
            print("Waiting for joint states...")
            return

        msg = Float32MultiArray()
        msg.data = [
            self._joint_state_subscriber.data.position[0],
            self._sim_joint_state_subscriber.data.position[0],
        ]

        print(f"Publishing: {msg.data}")

        self._pub.publish(msg)


def main():
    rclpy.init(args=None)

    node = JustTestNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
