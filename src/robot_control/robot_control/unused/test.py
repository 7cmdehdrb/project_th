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

        if self._cnt > 2:
            return

        msg = JointTrajectory()

        msg.header = Header(stamp=self.get_clock().now().to_msg())
        msg.joint_names = self._joint_states.name

        joint_states = self._joint_states

        if self._cnt % 2 == 0:
            print("Even count, using joint trajectory points with position changes")
            msg.points = [
                # JointTrajectoryPoint(
                #     positions=self._joint_states.position,
                #     velocities=self._joint_states.velocity,
                #     time_from_start=BuiltinDuration(sec=0, nanosec=0),
                # ),
                JointTrajectoryPoint(
                    positions=[
                        joint_states.position[0],
                        joint_states.position[1],
                        joint_states.position[2],
                        joint_states.position[3],
                        joint_states.position[4],
                        joint_states.position[5]
                        + (
                            np.deg2rad(5) * 0.01
                        ),  # After 0.1 seconds, move the last joint by 10 degrees
                    ],
                    velocities=[
                        joint_states.velocity[0],
                        joint_states.velocity[1],
                        joint_states.velocity[2],
                        joint_states.velocity[3],
                        joint_states.velocity[4],
                        np.deg2rad(5),
                    ],
                    time_from_start=BuiltinDuration(sec=0, nanosec=int(1e9 * 0.01)),
                ),
                JointTrajectoryPoint(
                    positions=[
                        joint_states.position[0],
                        joint_states.position[1],
                        joint_states.position[2],
                        joint_states.position[3],
                        joint_states.position[4],
                        joint_states.position[5]
                        + (
                            np.deg2rad(5) * 0.1
                        ),  # After 0.1 seconds, move the last joint by 10 degrees
                    ],
                    velocities=[
                        joint_states.velocity[0],
                        joint_states.velocity[1],
                        joint_states.velocity[2],
                        joint_states.velocity[3],
                        joint_states.velocity[4],
                        np.deg2rad(5),
                    ],
                    time_from_start=BuiltinDuration(sec=0, nanosec=int(1e9 * 0.1)),
                ),
                JointTrajectoryPoint(
                    positions=[
                        joint_states.position[0],
                        joint_states.position[1],
                        joint_states.position[2],
                        joint_states.position[3],
                        joint_states.position[4],
                        joint_states.position[5]
                        + (np.deg2rad(5) * 0.1 * 5.0),  # After 0.5 seconds
                    ],
                    velocities=[
                        joint_states.velocity[0],
                        joint_states.velocity[1],
                        joint_states.velocity[2],
                        joint_states.velocity[3],
                        joint_states.velocity[4],
                        np.deg2rad(5),
                    ],
                    time_from_start=BuiltinDuration(sec=0, nanosec=int(1e9 * 0.5)),
                ),
                JointTrajectoryPoint(
                    positions=[
                        joint_states.position[0],
                        joint_states.position[1],
                        joint_states.position[2],
                        joint_states.position[3],
                        joint_states.position[4],
                        joint_states.position[5]
                        + np.deg2rad(5) * 0.1 * 10.0,  # After 0.5 seconds
                    ],
                    velocities=[0.0] * 6,
                    time_from_start=BuiltinDuration(sec=1, nanosec=0),
                ),
            ]

        else:
            msg.points = [
                # JointTrajectoryPoint(
                #     positions=[
                #         self._joint_states.position[0],
                #         self._joint_states.position[1],
                #         self._joint_states.position[2],
                #         self._joint_states.position[3],
                #         self._joint_states.position[4],
                #         self._joint_states.position[5],
                #     ],
                #     velocities=self._joint_states.velocity,
                #     time_from_start=BuiltinDuration(sec=0, nanosec=0),
                # ),
                JointTrajectoryPoint(
                    positions=[
                        self._joint_states.position[0],
                        self._joint_states.position[1],
                        self._joint_states.position[2],
                        self._joint_states.position[3],
                        self._joint_states.position[4] + np.deg2rad(10) / 10.0,
                        self._joint_states.position[5],
                    ],
                    velocities=[
                        self._joint_states.velocity[0],
                        self._joint_states.velocity[1],
                        self._joint_states.velocity[2],
                        self._joint_states.velocity[3],
                        np.deg2rad(10),
                        self._joint_states.velocity[5],
                    ],
                    time_from_start=BuiltinDuration(sec=1),
                ),
                JointTrajectoryPoint(
                    positions=[
                        self._joint_states.position[0],
                        self._joint_states.position[1],
                        self._joint_states.position[2],
                        self._joint_states.position[3],
                        self._joint_states.position[4] + np.deg2rad(10) / 10.0 * 2.0,
                        self._joint_states.position[5],
                    ],
                    velocities=[
                        self._joint_states.velocity[0],
                        self._joint_states.velocity[1],
                        self._joint_states.velocity[2],
                        self._joint_states.velocity[3],
                        np.deg2rad(10),
                        self._joint_states.velocity[5],
                    ],
                    time_from_start=BuiltinDuration(sec=2),
                ),
            ]

        print("Publishing JointTrajectory message", np.deg2rad(5))

        # self._cnt += 1

        self._pub2.publish(msg)


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
            node.run2()
            rate.sleep()

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
