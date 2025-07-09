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
import os
import numpy as np
from base_package.manager import SimpleSubscriberManager


class TextLogger(object):
    def __init__(self, filename: str):
        self._filename = filename

        if os.path.exists(self._filename):
            raise FileExistsError(f"File '{self._filename}' already exists.")

        self._file = open(self._filename, "w")

        header_T = "Time,Stiffness,Damping,"
        header_R = "R_shoulder_pan_p,R_shoulder_lift_p,R_elbow_p,R_wrist_1_p,R_wrist_2_p,R_wrist_3_p,R_shoulder_pan_v,R_shoulder_lift_v,R_elbow_v,R_wrist_1_v,R_wrist_2_v,R_wrist_3_v,"
        header_S = "S_shoulder_pan_p,S_shoulder_lift_p,S_elbow_p,S_wrist_1_p,S_wrist_2_p,S_wrist_3_p,S_shoulder_pan_v,S_shoulder_lift_v,S_elbow_v,S_wrist_1_v,S_wrist_2_v,S_wrist_3_v,"
        header_C = "C_shoulder_pan_p,C_shoulder_lift_p,C_elbow_p,C_wrist_1_p,C_wrist_2_p,C_wrist_3_p"
        header = header_T + header_R + header_S + header_C

        self.log(header)

    def log(self, message: str):
        self._file.write(message + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


class RTDELogger(Node):
    def __init__(self):
        super().__init__("rtde_logger")

        self._text_logger = TextLogger("rtde_log_wrist3_zz.csv")

        self._real_joint_state_subscriber = SimpleSubscriberManager(
            self, "/joint_states", JointState
        )
        self._sim_joint_state_subscriber = SimpleSubscriberManager(
            self, "/isaac_sim/joint_states", JointState
        )
        self._control_command_subscriber = SimpleSubscriberManager(
            self, "/isaac_sim/joint_control", JointState
        )
        self._stiffness_subscriber = SimpleSubscriberManager(
            self, "/isaac_sim/stiffness", Float32MultiArray
        )
        self._damping_subscriber = SimpleSubscriberManager(
            self, "/isaac_sim/damping", Float32MultiArray
        )

        self._start_time = self.get_clock().now()

    def is_valid(self) -> bool:
        """
        Check if the logger is valid.
        :return: True if valid, False otherwise.
        """
        if self._real_joint_state_subscriber.data is None:
            self.get_logger().error("Real joint state subscriber is not initialized.")
            return False
        if self._sim_joint_state_subscriber.data is None:
            self.get_logger().error("Sim joint state subscriber is not initialized.")
            return False
        if self._control_command_subscriber.data is None:
            self.get_logger().error("Control command subscriber is not initialized.")
            return False
        if self._stiffness_subscriber.data is None:
            self.get_logger().error("Stiffness subscriber is not initialized.")
            return False
        if self._damping_subscriber.data is None:
            self.get_logger().error("Damping subscriber is not initialized.")
            return False
        self.get_logger().info("RTDE Logger is valid.")
        return True

    def log_data(self):
        """
        Log the data from the subscribers.
        """
        if not self.is_valid():
            return

        current_time: Time = self.get_clock().now()
        elapsed_time: Duration = current_time - self._start_time
        elapsed_time_sec = (
            elapsed_time.nanoseconds / 1e9
        )  # Convert nanoseconds to seconds

        real_joint_data: JointState = self._real_joint_state_subscriber.data
        sim_joint_data: JointState = self._sim_joint_state_subscriber.data
        control_command_data: JointState = self._control_command_subscriber.data
        stiffness_data: Float32MultiArray = self._stiffness_subscriber.data
        damping_data: Float32MultiArray = self._damping_subscriber.data

        time_text = f"{elapsed_time_sec:.3f}, {stiffness_data.data[0]}, {damping_data.data[0]}"  # Format time to 3 decimal places

        real_joint_position_text = f"{real_joint_data.position[5]}, {real_joint_data.position[0]}, {real_joint_data.position[1]}, {real_joint_data.position[2]}, {real_joint_data.position[3]}, {real_joint_data.position[4]}"
        real_joint_velocity_text = f"{real_joint_data.velocity[5]}, {real_joint_data.velocity[0]}, {real_joint_data.velocity[1]}, {real_joint_data.velocity[2]}, {real_joint_data.velocity[3]}, {real_joint_data.velocity[4]}"
        sim_joint_position_text = f"{sim_joint_data.position[5]}, {sim_joint_data.position[0]}, {sim_joint_data.position[1]}, {sim_joint_data.position[2]}, {sim_joint_data.position[3]}, {sim_joint_data.position[4]}"
        sim_joint_velocity_text = f"{sim_joint_data.velocity[5]}, {sim_joint_data.velocity[0]}, {sim_joint_data.velocity[1]}, {sim_joint_data.velocity[2]}, {sim_joint_data.velocity[3]}, {sim_joint_data.velocity[4]}"
        control_command_text = f"{control_command_data.position[0]}, {control_command_data.position[1]}, {control_command_data.position[2]}, {control_command_data.position[3]}, {control_command_data.position[4]}, {control_command_data.position[5]}"

        current_stiffness = int(stiffness_data.data[0])
        current_damping = int(damping_data.data[0])
        if current_stiffness == -1 and current_damping == -1:
            self.get_logger().info("Received end signal, closing logger.")
            self._text_logger.close()
            self.destroy_node()
            rclpy.shutdown()

        log_msg = (
            time_text
            + ", "
            + real_joint_position_text
            + ", "
            + real_joint_velocity_text
            + ", "
            + sim_joint_position_text
            + ", "
            + sim_joint_velocity_text
            + ", "
            + control_command_text
        ).replace(" ", "")

        self._text_logger.log(log_msg)


def main():

    rclpy.init(args=None)

    import threading

    node = RTDELogger()

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    rate = node.create_rate(10.0)  # 10 Hz

    while rclpy.ok():
        node.log_data()
        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()

    thread.join()


if __name__ == "__main__":
    main()
