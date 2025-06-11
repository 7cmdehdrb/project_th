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
import rtde_control as rtde_c
import rtde_receive as rtde_r
import time


class RobotControlNode(Node):
    def __init__(self):
        super().__init__("robot_control_node")

        # Initialize RTDE connection
        self.rtde_c = rtde_c.RTDEControlInterface(hostname="192.168.56.101")
        self.rtde_r = rtde_r.RTDEReceiveInterface(hostname="192.168.56.101")

        self._joint_states_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            qos_profile=qos_profile_system_default,
        )
        self._joint_state: JointState = None

        self._t1 = time.time()

    def _joint_state_callback(self, msg: JointState):
        self._joint_state = msg

    def run(self):
        if self._joint_state is None:
            self.get_logger().warn("Joint states not received yet.")
            return

        current_time = time.time()
        dt = current_time - self._t1

        print(f"Time since last callback: {dt:.2f} seconds")

        self._t1 = current_time

        new_joint = [
            self._joint_state.position[5],
            self._joint_state.position[0],
            self._joint_state.position[1],
            self._joint_state.position[2],
            self._joint_state.position[3],
            self._joint_state.position[4],
        ]
        new_joint[0] += -np.deg2rad(9) / 10.0 * 2.0

        self.rtde_c.moveJ(new_joint, asynchronous=True)


def main():
    rclpy.init(args=None)

    node = RobotControlNode()

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 10.0  # Frequency in Hz
    rate = node.create_rate(hz)

    try:
        while rclpy.ok():
            node.run()
            rate.sleep()

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
