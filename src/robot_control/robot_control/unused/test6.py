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


class TextLogger(object):
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "w")

    def log(self, message):
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")

        self._joint_state_manager = SimpleSubscriberManager(
            self, "joint_states", JointState, qos_profile=qos_profile_system_default
        )

        self._text_logger = TextLogger("test_log.txt")
        self._timer = self.create_timer(0.1, self._run)

    def _run(self):
        if self._joint_state_manager.data is None:
            self.get_logger().warn("Joint states not received yet.")
            return None

        joint_states: JointState = self._joint_state_manager.data
        joint_positions = joint_states.position
        log_text = f"{joint_positions[5]},{joint_positions[0]},{joint_positions[1]},{joint_positions[2]},{joint_positions[3]},{joint_positions[4]}"
        self._text_logger.log(log_text)


def main():
    rclpy.init(args=None)

    import threading

    node = TestNode()

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    while rclpy.ok():
        pass

    node._text_logger.close()

    rclpy.shutdown()
    node.destroy_node()

    thread.join()


if __name__ == "__main__":
    main()
