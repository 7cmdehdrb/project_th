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
import socket


class SocketServer:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8888

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

    def listen(self):
        """
        Listen for a single 7-float (28 bytes) packet from a client.
        Returns:
            list[float]: The received 7 floats, or None if connection closed.
        """
        conn, addr = self.server_socket.accept()
        try:
            data = b""
            while len(data) < 28:
                packet = conn.recv(28 - len(data))
                if not packet:
                    return None
                data += packet
            floats = np.frombuffer(data, dtype=np.float32)
            return floats.tolist()
        finally:
            conn.close()


class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")

        self._socket_server = SocketServer()
        self.get_logger().info("Socket server initialized, waiting for connections...")

        self._default_joint = np.array(
            [
                0.0,
                -2.2,
                2.2,
                0.0,
                1.57,
                np.pi / 2.0,
            ]
        )

        self._control_joint_states_publisher = self.create_publisher(
            JointState,
            "/isaac_sim/joint_control",
            qos_profile=qos_profile_system_default,
        )
        self._damping_publisher = self.create_publisher(
            Float32MultiArray,
            "isaac_sim/damping",
            qos_profile=qos_profile_system_default,
        )
        self._stiffness_publisher = self.create_publisher(
            Float32MultiArray,
            "isaac_sim/stiffness",
            qos_profile=qos_profile_system_default,
        )

    def initialize(self):
        rate = self.create_rate(100)  # 10 Hz
        for _ in range(10):
            self._control_joint_states_publisher.publish(
                JointState(
                    position=self._default_joint.tolist(),  # Convert numpy array to list
                )
            )
            rate.sleep()

    def run(self):
        q = self._socket_server.listen()
        cmd = self._default_joint + np.array(q[:6], dtype=np.float32) * 0.5

        self._stiffness_publisher.publish(
            Float32MultiArray(
                data=[261.0] * 6
            )  # [600.0, 1000.0, 1000.0, 1000.0, 600.0, 600.0]
        )
        self._damping_publisher.publish(
            Float32MultiArray(
                data=[26.1] * 6
            )  # [40.0, 100.0, 100.0, 100.0, 70.0, 70.0]
        )
        self._control_joint_states_publisher.publish(
            JointState(
                position=cmd.tolist(),  # Convert numpy array to list
            )
        )


def main():
    rclpy.init(args=None)

    node = TestNode()

    import threading

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    rate = node.create_rate(100)  # 10 Hz

    node.initialize()

    while rclpy.ok():
        try:
            node.run()
        except Exception as e:
            node.get_logger().error(f"Error in run: {e}")

        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()

    th.join()


if __name__ == "__main__":
    main()
