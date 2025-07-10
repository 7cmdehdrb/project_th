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
import pandas as pd
import rtde_control as rtde_c
import rtde_receive as rtde_r
from base_package.manager import SimpleSubscriberManager
from enum import Enum


class PathManager(object):
    def __init__(self, filename):
        self._filename = filename

        # self._data = pd.read_csv(self._filename)
        # self._path = self._data.to_numpy()

        self._data = pd.read_csv(self._filename)[
            ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
        ]
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
        self._path = self._data.to_numpy()

    def get_entire_path(self) -> np.ndarray:
        """
        Get the entire path as a numpy array.
        :return: Numpy array representing the entire path.
        """
        return self._path

    def get_path(self, idx: int) -> np.ndarray:
        """
        Get the path at the specified index.
        :param idx: Index of the path to retrieve.
        :return: Numpy array representing the path.
        """
        if idx < 0 or idx >= len(self._path):
            raise IndexError("Index out of bounds for path data.")

        return self._path[idx] + self._default_joint


class RTDEGainTuner(object):
    def __init__(self, node: Node):
        self._node = node

        self._stiffness_publisher = self._node.create_publisher(
            Float32MultiArray,
            "isaac_sim/stiffness",
            qos_profile=qos_profile_system_default,
        )
        self._damping_publisher = self._node.create_publisher(
            Float32MultiArray,
            "isaac_sim/damping",
            qos_profile=qos_profile_system_default,
        )


class Mode(Enum):
    SHOULDERPAN = 0
    SHOULDERLIFT = 1
    ELBOW = 2
    WRIST1 = 3
    WRIST2 = 4
    WRIST3 = 5


class RTDEControlNode(Node):
    def __init__(self):
        super().__init__("rtde_control_node")

        IP = "192.168.56.101"

        # Initialize RTDE control and receive interfaces
        self._rtde_control = rtde_c.RTDEControlInterface(IP)
        self._rtde_receive = rtde_r.RTDEReceiveInterface(IP)

        # Initialize gain tuner and path manager
        self._joint_state_subscriber = SimpleSubscriberManager(
            self, "/joint_states", JointState
        )
        self._gain_tuner = RTDEGainTuner(self)
        self._path_manager = PathManager(
            "/home/min/7cmdehdrb/project_th/data[2by2].csv"
        )

        self._loop_once = True  # Flag to control the loop execution
        self._clockwise = True  # Flag to control the direction of the path
        self._mode = Mode.ELBOW  # Default mode

        # # Scenario: 0
        # # Define stiffness and damping values
        # self._stiffnesses = np.linspace(600.0, 1500.0, num=20)
        # self._dampings = np.linspace(40.0, 100.0, num=20)

        # Scenario: 1
        # Define stiffness and damping values
        self._stiffnesses = np.linspace(600.0, 1800.0, num=10)
        self._dampings = np.linspace(30.0, 180.0, num=10)

        # # MAX
        # self._stiffnesses = np.linspace(1405.26318359375, 1405.26318359375, num=1)
        # self._dampings = np.linspace(52.6315803527832, 52.6315803527832, num=1)

        # # MIN
        # self._stiffnesses = np.linspace(884.2105102539062, 884.2105102539062, num=1)
        # self._dampings = np.linspace(77.89473724365234, 77.89473724365234, num=1)

        # Create publishers
        self._joint_state_publisher = self.create_publisher(
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

    def run2(self):
        if self._joint_state_subscriber.data is None:
            self.get_logger().error("Joint state subscriber is not initialized.")
            return

        rate = self.create_rate(100.0)

        for i, damping in enumerate(self._dampings):
            if i % 2 == 0:
                stiffnesses_iter = self._stiffnesses
            else:
                stiffnesses_iter = self._stiffnesses[::-1]

            for stiffness in stiffnesses_iter:

                self.get_logger().info(
                    f"Starting control loop with stiffness: {stiffness}, damping: {damping}"
                )

                while rclpy.ok():
                    joint_states: JointState = self._joint_state_subscriber.data

                    shoulder_pan = joint_states.position[5]
                    shoulder_lift = joint_states.position[0]
                    elbow = joint_states.position[1]
                    wrist_1 = joint_states.position[2]
                    wrist_2 = joint_states.position[3]
                    wrist_3 = joint_states.position[4]

                    control_command = [
                        shoulder_pan,
                        shoulder_lift,
                        elbow,
                        wrist_1,
                        wrist_2,
                        wrist_3,
                    ]

                    control_command[self._mode.value] += np.random.uniform(
                        0.1, np.deg2rad(45.0)
                    ) * (1.0 if self._clockwise else -1.0)

                    self._rtde_control.servoJ(
                        control_command,
                        0.0,
                        0.0,
                        0.01,
                        0.2,
                        300.0,
                    )

                    # 2. Publish the Joint State for Isaac Sim
                    msg = JointState(
                        position=control_command,
                    )
                    self._joint_state_publisher.publish(msg)

                    # Set stiffness and damping values
                    stiffness_msg = Float32MultiArray(data=[stiffness] * 6)
                    damping_msg = Float32MultiArray(data=[damping] * 6)

                    self._stiffness_publisher.publish(stiffness_msg)
                    self._damping_publisher.publish(damping_msg)

                    # print(f"Published Stiffness: {stiffness}, Damping: {damping}")

                    if (
                        self._clockwise
                        and np.rad2deg(control_command[self._mode.value]) > 170.0
                    ) or (
                        not self._clockwise
                        and np.rad2deg(control_command[self._mode.value]) < -170.0
                    ):
                        self._clockwise = not self._clockwise
                        break

                    rate.sleep()

        if self._loop_once:
            self.get_logger().info("Loop completed once, shutting down.")

            for _ in range(30):
                stiffness_msg = Float32MultiArray(data=[-1] * 6)
                damping_msg = Float32MultiArray(data=[-1] * 6)

                self._stiffness_publisher.publish(stiffness_msg)
                self._damping_publisher.publish(damping_msg)

                rate.sleep()

            rclpy.shutdown()

    def run(self):
        rate = self.create_rate(100.0)  # 10 Hz

        # Iterate over stiffness and damping values
        for stiffness in self._stiffnesses:
            for damping in self._dampings:

                # Set stiffness and damping values
                for idx in range(len(self._path_manager.get_entire_path())):
                    for _ in range(10):  # Repeat each path point 10 times
                        # 1. Control the robot to follow the path
                        self._rtde_control.servoJ(
                            self._path_manager.get_path(idx).tolist(),
                            0.0,
                            0.0,
                            0.01,
                            0.2,
                            300.0,
                        )

                        print(
                            f"Moving to path point {idx} with stiffness {stiffness} and damping {damping}"
                        )

                        # 2. Publish the Joint State for Isaac Sim
                        msg = JointState(
                            position=self._path_manager.get_path(idx).tolist(),
                        )
                        self._joint_state_publisher.publish(msg)

                        # Publish stiffness and damping values
                        stiffness_msg = Float32MultiArray(data=[stiffness] * 6)
                        damping_msg = Float32MultiArray(data=[damping] * 6)

                        self._stiffness_publisher.publish(stiffness_msg)
                        self._damping_publisher.publish(damping_msg)

                        self.get_logger().info(
                            f"Published Stiffness: {stiffness}, Damping: {damping}"
                        )

                        rate.sleep()


def main():
    rclpy.init(args=None)

    import threading

    node = RTDEControlNode()

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    while rclpy.ok():
        node.run2()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
