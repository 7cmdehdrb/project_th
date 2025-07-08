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
from trajectory_msgs.msg import *
from moveit_msgs.msg import *

# TF
from tf2_ros import *

# Custom Package
from base_package.manager import SimpleSubscriberManager
from moveit2_commander import (
    FK_ServiceManager,
    IK_ServiceManager,
    KinematicPath_ServiceManager,
    CartesianPath_ServiceManager,
    ExecuteTrajectory_ServiceManager,
)

# Python
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple, List


class FileLogger(object):
    def __init__(self, node: Node):
        self._node = node

        self._f = open("REMOVE.csv", "w")

    def log(self, message: str):
        """
        Log a message to the log file and the console.
        """
        self._f.write(message + "\n")
        self._f.flush()

    def __del__(self):
        """
        Close the log file when the Logger object is deleted.
        """
        if self._f:
            self._f.close()
            self._node.get_logger().info("Logger closed and file saved.")
        else:
            self._node.get_logger().warn("Logger file was not open.")


class GainTuner(Node):
    def __init__(self):
        super().__init__("gain_tuner")

        # >>> Joint State Management >>>
        self._joint_state_sub = SimpleSubscriberManager(
            self, "joint_states", JointState, qos_profile=qos_profile_system_default
        )
        self._shoulder_pan_sub = SimpleSubscriberManager(
            self,
            "/joint_feedbacks/shoulder_pan",
            Float32MultiArray,
            qos_profile=qos_profile_system_default,
        )
        self._shoulder_lift_sub = SimpleSubscriberManager(
            self,
            "/joint_feedbacks/shoulder_lift",
            Float32MultiArray,
            qos_profile=qos_profile_system_default,
        )
        self._elbow_sub = SimpleSubscriberManager(
            self,
            "/joint_feedbacks/elbow",
            Float32MultiArray,
            qos_profile=qos_profile_system_default,
        )
        self._wrist_1_sub = SimpleSubscriberManager(
            self,
            "/joint_feedbacks/wrist1",
            Float32MultiArray,
            qos_profile=qos_profile_system_default,
        )
        self._wrist_2_sub = SimpleSubscriberManager(
            self,
            "/joint_feedbacks/wrist2",
            Float32MultiArray,
            qos_profile=qos_profile_system_default,
        )
        self._wrist_3_sub = SimpleSubscriberManager(
            self,
            "/joint_feedbacks/wrist3",
            Float32MultiArray,
            qos_profile=qos_profile_system_default,
        )
        # <<< Joint State Management <<<

        self._exp_num_sub = SimpleSubscriberManager(
            self, "experiment_number", UInt32, qos_profile=qos_profile_system_default
        )

        self._file_logger = FileLogger(self)

        header = "EXP, ShoulderPan_PE, ShoulderPan_VE, ShoulderLift_PE, \ShoulderLift_VE, Elbow_PE, \
            Elbow_VE, Wrist1_PE, Wrist1_VE, Wrist2_PE, Wrist2_VE, Wrist3_PE, Wrist3_VE"
        self._file_logger.log(header)

    def run(self):
        self._compare_joint_states()  # This method can be used to run any additional logic if needed

    def _compare_joint_states(self):
        if not self._check_data_validation():
            return None

        # Define the simulated joint data
        sim_shoulder_pan = self._shoulder_pan_sub.data
        sim_shoulder_lift = self._shoulder_lift_sub.data
        sim_elbow = self._elbow_sub.data
        sim_wrist_1 = self._wrist_1_sub.data
        sim_wrist_2 = self._wrist_2_sub.data
        sim_wrist_3 = self._wrist_3_sub.data

        # Define the real joint positions and velocities
        real_shoulder_pan_position = self._joint_state_sub.data.position[5]
        real_shoulder_list_position = self._joint_state_sub.data.position[0]
        real_elbow_position = self._joint_state_sub.data.position[1]
        real_wrist_1_position = self._joint_state_sub.data.position[2]
        real_wrist_2_position = self._joint_state_sub.data.position[3]
        real_wrist_3_position = self._joint_state_sub.data.position[4]

        real_shoulder_pan_velocity = self._joint_state_sub.data.velocity[5]
        real_shoulder_list_velocity = self._joint_state_sub.data.velocity[0]
        real_elbow_velocity = self._joint_state_sub.data.velocity[1]
        real_wrist_1_velocity = self._joint_state_sub.data.velocity[2]
        real_wrist_2_velocity = self._joint_state_sub.data.velocity[3]
        real_wrist_3_velocity = self._joint_state_sub.data.velocity[4]

        # Compare the simulated joint data with the real joint positions and velocities
        shoulder_pan_p_error, shoulder_pan_v_error = self._compare_joint(
            sim_shoulder_pan,
            real_shoulder_pan_position,
            real_shoulder_pan_velocity,
        )
        shoulder_lift_p_error, shoulder_lift_v_error = self._compare_joint(
            sim_shoulder_lift,
            real_shoulder_list_position,
            real_shoulder_list_velocity,
        )
        elbow_p_error, elbow_v_error = self._compare_joint(
            sim_elbow,
            real_elbow_position,
            real_elbow_velocity,
        )
        wrist_1_p_error, wrist_1_v_error = self._compare_joint(
            sim_wrist_1,
            real_wrist_1_position,
            real_wrist_1_velocity,
        )
        wrist_2_p_error, wrist_2_v_error = self._compare_joint(
            sim_wrist_2,
            real_wrist_2_position,
            real_wrist_2_velocity,
        )
        wrist_3_p_error, wrist_3_v_error = self._compare_joint(
            sim_wrist_3,
            real_wrist_3_position,
            real_wrist_3_velocity,
        )

        print("Joint Position and Velocity Errors:")
        print(
            f"Shoulder Pan: PE: {shoulder_pan_p_error:.4f}, VE: {shoulder_pan_v_error:.4f}"
        )
        print(
            f"Shoulder Lift: PE: {shoulder_lift_p_error:.4f}, VE: {shoulder_lift_v_error:.4f}"
        )
        print(f"Elbow: PE: {elbow_p_error:.4f}, VE: {elbow_v_error:.4f}")
        print(f"Wrist 1: PE: {wrist_1_p_error:.4f}, VE: {wrist_1_v_error:.4f}")
        print(f"Wrist 2: PE: {wrist_2_p_error:.4f}, VE: {wrist_2_v_error:.4f}")
        print(f"Wrist 3: PE: {wrist_3_p_error:.4f}, VE: {wrist_3_v_error:.4f}")
        print()

        exp_num: UInt32 = self._exp_num_sub.data

        # Log the errors to the file
        log_message = (
            f"{int(exp_num.data)}, {shoulder_pan_p_error:.4f}, {shoulder_pan_v_error:.4f}, "
            f"{shoulder_lift_p_error:.4f}, {shoulder_lift_v_error:.4f}, "
            f"{elbow_p_error:.4f}, {elbow_v_error:.4f}, "
            f"{wrist_1_p_error:.4f}, {wrist_1_v_error:.4f}, "
            f"{wrist_2_p_error:.4f}, {wrist_2_v_error:.4f}, "
            f"{wrist_3_p_error:.4f}, {wrist_3_v_error:.4f}"
        )
        self.get_logger().info("Logging...")
        self._file_logger.log(log_message)

    def _compare_joint(
        self,
        sim_joint_data: Float32MultiArray,
        real_joint_position: float,
        real_joint_velocity: float,
    ) -> Tuple[float, float]:
        """
        Compare the simulated joint data with the real joint position and velocity.

        Returns the position and velocity errors.
        """
        sim_joint_position = np.deg2rad(sim_joint_data.data[0])
        sim_joint_velocity = np.deg2rad(sim_joint_data.data[1])

        real_joint_position = real_joint_position
        real_joint_velocity = real_joint_velocity

        position_error = float(sim_joint_position - real_joint_position)
        velocity_error = float(sim_joint_velocity - real_joint_velocity)

        return position_error, velocity_error

    def _check_data_validation(self):
        # This method can be used to validate the data received from the subscribers
        if not self._exp_num_sub.data:
            self.get_logger().warn("Experiment Number Data is empty")
            return False
        if not self._joint_state_sub.data:
            self.get_logger().warn("Joint State Data is empty")
            return False
        if not self._shoulder_pan_sub.data:
            self.get_logger().warn("Shoulder Pan Data is empty")
            return False
        if not self._shoulder_lift_sub.data:
            self.get_logger().warn("Shoulder Lift Data is empty")
            return False
        if not self._elbow_sub.data:
            self.get_logger().warn("Elbow Data is empty")
            return False
        if not self._wrist_1_sub.data:
            self.get_logger().warn("Wrist 1 Data is empty")
            return False
        if not self._wrist_2_sub.data:
            self.get_logger().warn("Wrist 2 Data is empty")
            return False
        if not self._wrist_3_sub.data:
            self.get_logger().warn("Wrist 3 Data is empty")
            return False
        return True


def main():

    rclpy.init(args=None)

    node = GainTuner()

    import threading

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 10.0
    rate = node.create_rate(hz)

    while rclpy.ok():
        node.run()
        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()

    thread.join()


if __name__ == "__main__":
    main()
