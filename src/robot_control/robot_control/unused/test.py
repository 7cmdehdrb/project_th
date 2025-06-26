# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from builtin_interfaces.msg import Duration as BuiltinDuration
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from control_msgs.msg import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# TF
from tf2_ros import *

# Python
import numpy as np


class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")

        # Example of creating a publisher
        self._pub1 = self.create_publisher(
            JointJog,
            "/servo_node/delta_joint_cmds",
            qos_profile=qos_profile_system_default,
        )
        self._pub4 = self.create_publisher(
            TwistStamped,
            "/servo_node/delta_twist_cmds",
            qos_profile=qos_profile_system_default,
        )
        self._pub2 = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            qos_profile=qos_profile_system_default,
        )
        self._pub3 = self.create_publisher(
            Float64MultiArray,
            "/forward_velocity_controller/commands",
            qos_profile=qos_profile_system_default,
        )

        self._subscriber = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            qos_profile=qos_profile_system_default,
        )
        self._joint_states: JointState = None

        self._cnt = 0

    def _joint_state_callback(self, msg: JointState):
        self._joint_states = msg

    def run2(self):
        if self._joint_states is None:
            return

        msg = JointJog()

        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="base_link")
        msg.joint_names = self._joint_states.name

        joint_states = self._joint_states

        msg.displacements = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01]
        msg.velocities = joint_states.velocity

        msg.velocities[5] = -1.0

        print(f"Target joint velocities: {msg.velocities}")
        print(f"Current joint velocities: {joint_states.velocity}")

        msg.duration = 0.1

        # print("Publishing JointJog message")

        self._pub1.publish(msg)

    def run3(self):
        if self._joint_states is None:
            return

        msg = TwistStamped()

        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="base_link")

        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.01
        msg.twist.linear.z = 0.0

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        self._pub4.publish(msg)

    def run(self):
        if self._joint_states is None:
            return

        msg = JointTrajectory()

        msg.header = Header(stamp=self.get_clock().now().to_msg())
        msg.joint_names = self._joint_states.name

        joint_states = self._joint_states

        msg.points = [
            JointTrajectoryPoint(
                positions=self._joint_states.position,
                velocities=self._joint_states.velocity,
                time_from_start=BuiltinDuration(sec=0, nanosec=0),
            ),
            JointTrajectoryPoint(
                # positions=[
                #     joint_states.position[0],
                #     joint_states.position[1],
                #     joint_states.position[2],
                #     joint_states.position[3],
                #     joint_states.position[4],
                #     joint_states.position[5]
                #     + (
                #         np.deg2rad(30) * 0.1
                #     ),  # After 0.1 seconds, move the last joint by 5 degrees
                # ],
                velocities=[
                    joint_states.velocity[0],
                    joint_states.velocity[1],
                    joint_states.velocity[2],
                    joint_states.velocity[3],
                    joint_states.velocity[4],
                    np.deg2rad(30),
                ],
                time_from_start=BuiltinDuration(sec=0, nanosec=int(1e9 * 0.1)),
            ),
            JointTrajectoryPoint(
                # positions=[
                #     joint_states.position[0],
                #     joint_states.position[1],
                #     joint_states.position[2],
                #     joint_states.position[3],
                #     joint_states.position[4],
                #     joint_states.position[5]
                #     + (np.deg2rad(30) * 0.1 * 5.0),  # After 0.5 seconds
                # ],
                velocities=[
                    joint_states.velocity[0],
                    joint_states.velocity[1],
                    joint_states.velocity[2],
                    joint_states.velocity[3],
                    joint_states.velocity[4],
                    np.deg2rad(30),
                ],
                time_from_start=BuiltinDuration(sec=0, nanosec=int(1e9 * 0.5)),
            ),
            JointTrajectoryPoint(
                # positions=[
                #     joint_states.position[0],
                #     joint_states.position[1],
                #     joint_states.position[2],
                #     joint_states.position[3],
                #     joint_states.position[4],
                #     joint_states.position[5]
                #     + np.deg2rad(30) * 0.1 * 10.0,  # After 0.5 seconds
                # ],
                velocities=[0.0] * 6,
                time_from_start=BuiltinDuration(sec=1, nanosec=0),
            ),
        ]

        print("Publishing JointTrajectory message", np.deg2rad(30))

        self._pub2.publish(msg)

    def run4(self):
        if self._joint_states is None:
            return

        msg = Float64MultiArray()

        joint_states = self._joint_states

        msg.data = [0.00, 0.0, 0.0, 0.05, 0.0, 0.0]

        print("Publishing Float64MultiArray message")

        self._pub3.publish(msg)


def main():
    rclpy.init(args=None)

    node = TestNode()

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 10.0  # Frequency in Hz
    rate = node.create_rate(hz)

    try:
        while rclpy.ok():
            node.run4()
            rate.sleep()

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
