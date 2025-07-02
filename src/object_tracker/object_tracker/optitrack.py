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
import natnetclient as natnet


class OptiTrackNode(Node):
    def __init__(self):
        super().__init__("optitrack_node")

        # Initialize NatNet client
        IP = "127.0.0.1"
        DATA_PORT = 1511
        COMM_PORT = 1510

        self._client = natnet.NatClient(
            client_ip=IP, data_port=DATA_PORT, comm_port=COMM_PORT
        )


def main():
    rclpy.init(args=None)

    node = None

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
