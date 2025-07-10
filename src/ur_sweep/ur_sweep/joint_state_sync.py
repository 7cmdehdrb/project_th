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


def reorder_joint_states(joint_states: JointState) -> JointState:
    """
    Reorder joint states to match the UR5e DH parameter order.

    Args:
        joint_states (JointState): Original joint states.

    Returns:
        JointState: Reordered joint states.
    """
    reordered_positions = [
        joint_states.position[5],  # shoulder_pan_joint
        joint_states.position[0],  # shoulder_lift_joint
        joint_states.position[1],  # elbow_joint
        joint_states.position[2],  # wrist_1_joint
        joint_states.position[3],  # wrist_2_joint
        joint_states.position[4],  # wrist_3_joint
    ]
    reordered_velocities = [
        joint_states.velocity[5],
        joint_states.velocity[0],
        joint_states.velocity[1],
        joint_states.velocity[2],
        joint_states.velocity[3],
        joint_states.velocity[4],
    ]

    return JointState(position=reordered_positions, velocity=reordered_velocities)


class JointStateSyncro(Node):
    def __init__(self):
        super().__init__("joint_state_syncro")

        self._joint_state_sub = SimpleSubscriberManager(
            node=self,
            topic_name="/isaac_sim/joint_states",  # /isaac_sim/joint_states
            msg_type=JointState,
            qos_profile=qos_profile_system_default,
        )

        self._control_joint_states_publisher = self.create_publisher(
            JointState,
            "/isaac_sim/joint_control",
            qos_profile=qos_profile_system_default,
        )

    def run(self):
        if self._joint_state_sub.data is None:
            self.get_logger().warn("No joint states received yet.")
            return None

        joint_states: JointState = self._joint_state_sub.data
        joint_states = reorder_joint_states(joint_states)

        self._control_joint_states_publisher.publish(joint_states)


def main():
    rclpy.init(args=None)

    import threading

    node = JointStateSyncro()

    th = threading.Thread(target=rclpy.spin, args=(node,))
    th.start()

    rate = node.create_rate(100)  # 500 Hz
    while rclpy.ok():
        node.run()
        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()

    th.join()


if __name__ == "__main__":
    main()
