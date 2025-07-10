# Standard library imports
import math
import time

# Third-party imports
import numpy as np
import torch
import rotutils

# ROS2 core imports
import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, qos_profile_system_default

# ROS2 message imports
import rclpy.time
from std_msgs.msg import *
from geometry_msgs.msg import *
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from tf2_geometry_msgs import do_transform_pose
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# ROS2 TF imports
import tf2_ros
from tf2_ros import *

# UR RTDE imports
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# Local imports
from base_package.manager import SimpleSubscriberManager, TransformManager
from ur_sweep.ur_reach_policy import URReachPolicy
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


class SocketClient:
    def __init__(self, host="127.0.0.1", port=8888):
        self.host = host
        self.port = port

    def send_floats(self, floats):
        """
        Send a list of 7 floats to the server.
        Args:
            floats (list[float]): List of 7 floats to send.
        Returns:
            bool: True if sent successfully, False otherwise.
        """
        if len(floats) != 7:
            raise ValueError("Exactly 7 floats must be provided.")
        data = np.array(floats, dtype=np.float32).tobytes()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                s.sendall(data)
            return True
        except Exception as e:
            print(f"SocketClient send error: {e}")
            return False

    def receive_floats(self):
        """
        Connect to the server and receive a list of 7 floats.
        Returns:
            list[float] or None: Received floats, or None if failed.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                data = b""
                while len(data) < 28:
                    packet = s.recv(28 - len(data))
                    if not packet:
                        return None
                    data += packet
                floats = np.frombuffer(data, dtype=np.float32)
                return floats.tolist()
        except Exception as e:
            print(f"SocketClient receive error: {e}")
            return None


class EEFManager(object):
    def __init__(self, node: Node):
        self._node = node

        self._joint_states_manager = SimpleSubscriberManager(
            node=self._node,
            topic_name="/joint_states",
            msg_type=JointState,
        )

        self._dh_params = [
            (0, 0.1625, np.pi / 2),
            (-0.425, 0, 0),
            (-0.3922, 0, 0),
            (0, 0.1333, np.pi / 2),
            (0, 0.0997, -np.pi / 2),
            (0, 0.0996, 0),
        ]
        self._gripper_offset = np.array([0, 0, 0.12])  # Gripper offset in Z direction

    def dh_transform(self, a, d, alpha, theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1],
            ]
        )

    def forward_kinematics(self, joint_states: JointState | None) -> np.ndarray:
        if joint_states is None:
            joint_states: JointState = self._joint_states_manager.data

        if joint_states is None:
            self._node.get_logger().warn("JointState data not available yet.")
            return None

        # UR5e → DH 파라미터 순서로 재정렬
        q = np.array(
            [
                joint_states.position[5],
                joint_states.position[0],
                joint_states.position[1],
                joint_states.position[2],
                joint_states.position[3],
                joint_states.position[4],
            ]
        )

        # 누적 변환
        T = np.eye(4)
        for i in range(6):
            a, d, alpha = self._dh_params[i]
            T = T @ self.dh_transform(a, d, alpha, q[i])

        # 그리퍼(고정 오프셋) 적용
        T_grip = np.eye(4)
        T_grip[:3, 3] = self._gripper_offset
        T_eef = T @ T_grip

        T_correction = np.diag([-1, -1, 1, 1])  # x, y 반전

        T_eef_corrected = T_correction @ T_eef

        rotation_matrix = T_eef_corrected[:3, :3]
        qx, qy, qz, qw = rotutils.quaternion_from_rotation_matrix(
            rotation_matrix=rotation_matrix
        )

        return np.array(
            [
                T_eef_corrected[0, 3],
                T_eef_corrected[1, 3],
                T_eef_corrected[2, 3],
                qw,
                qx,
                qy,
                qz,
            ]
        )


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


def interpolate_joint_states(
    joint_states: JointState,
    current_time: Time,
):

    msg_time = joint_states.header.stamp

    dsec = current_time.to_msg().sec - msg_time.sec
    dnanosec = current_time.to_msg().nanosec - msg_time.nanosec

    dt = dsec + dnanosec * 1e-9  # Convert to seconds

    if dt < 0.01:
        # If the time difference is too small, return the original joint states
        return joint_states

    positions = np.array(joint_states.position)
    velocities = np.array(joint_states.velocity)

    interpolated_positions = positions + velocities * dt

    return JointState(
        header=joint_states.header,
        position=interpolated_positions.tolist(),
        velocity=velocities.tolist(),
        effort=joint_states.effort,
    )


class ReachPolicy(Node):
    """ROS2 node for controlling a UR robot's reach policy"""

    # Define simulation degree-of-freedom angle limits: (Lower limit, Upper limit, Inversed flag)
    SIM_DOF_ANGLE_LIMITS = [
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
    ]

    # Define servo angle limits (in radians)
    PI = np.pi
    SERVO_ANGLE_LIMITS = [
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
    ]

    # ROS topics and joint names
    STATE_TOPIC = "/scaled_joint_trajectory_controller/state"
    CMD_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
    JOINT_NAMES = [
        "elbow_joint",
        "shoulder_lift_joint",
        "shoulder_pan_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # Mapping from joint name to simulation action index
    JOINT_NAME_TO_IDX = {
        "elbow_joint": 2,
        "shoulder_lift_joint": 1,
        "shoulder_pan_joint": 0,
        "wrist_1_joint": 3,
        "wrist_2_joint": 4,
        "wrist_3_joint": 5,
    }

    def __init__(self):
        """Initialize the ReachPolicy node"""
        super().__init__("reach_policy_node")

        # TODO: Specify the paths to your model and YAML files
        self.robot = URReachPolicy(
            node=self,
            model_file="/home/min/7cmdehdrb/project_th/src/ur_sweep/resource/7_set/exported/policy.pt",  # Specify the path to your model file
            yaml_file="/home/min/7cmdehdrb/project_th/src/ur_sweep/resource/7_set/params/env.yaml",  # Specify the path to your model and YAML file
        )

        # RTDE
        IP = "192.168.56.101"
        self._rtde_c = RTDEControlInterface(IP)
        self._rtde_r = RTDEReceiveInterface(IP)

        self._initialized = False
        self._tf_manager = TransformManager(node=self)
        self._eef_manager = EEFManager(node=self)
        self._sim_object_manager = SimpleSubscriberManager(
            node=self,
            topic_name="/isaac_sim/pose/mug_a",
            msg_type=PoseStamped,
        )
        self._joint_state_manager = SimpleSubscriberManager(
            node=self,
            topic_name="/joint_states",  # /isaac_sim/joint_states
            msg_type=JointState,
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

        self._reset()

    def run(self):
        joint_states: JointState = self._joint_state_manager.data
        if self._joint_state_manager.data is None:
            error_msg = "JointState data not available yet. Skipping iteration."
            raise ValueError(error_msg)

        # joint_states = interpolate_joint_states(
        #     joint_states=joint_states, current_time=self.get_clock().now()
        # )

        tcp_pose = self._eef_manager.forward_kinematics(joint_states=joint_states)
        if tcp_pose is None:
            error_msg = "TCP pose is not available. Skipping servo command."
            raise ValueError(error_msg)

        self._stiffness_publisher.publish(
            Float32MultiArray(
                data=[600.0, 1000.0, 1000.0, 1000.0, 600.0, 600.0]
            )  # [600.0, 1000.0, 1000.0, 1000.0, 600.0, 600.0]
        )
        self._damping_publisher.publish(
            Float32MultiArray(
                data=[40.0, 100.0, 100.0, 100.0, 70.0, 70.0]
            )  # [40.0, 100.0, 100.0, 100.0, 70.0, 70.0]
        )

        # STEP 1: Get the current joint positions and velocities
        # current_pos = self._rtde_r.getActualQ()
        # current_vel = self._rtde_r.getActualQd()
        joint_states: JointState = reorder_joint_states(joint_states)

        current_pos = joint_states.position
        current_vel = joint_states.velocity

        # STEP 2: Get the current TCP pose in base_link frame
        # Update the robot's joint state
        self.robot.update_joint_state(current_pos, current_vel)
        # Update the robot's TCP state

        self.robot.update_tcp_state(pose=tcp_pose)  # tcp_pose

        # Update the object pose in the simulation
        object_pose: PoseStamped = self._sim_object_manager.data
        if object_pose is None:
            error_msg = "Object pose data not available yet. Skipping iteration."
            raise ValueError(error_msg)

        self.robot.update_target_state(
            pos=[
                object_pose.pose.position.x,
                object_pose.pose.position.y,
                object_pose.pose.position.z,
            ]
        )
        if self._initialized is False:  # self._initialized is False:
            self.robot.update_goal_state(
                goal=[
                    object_pose.pose.position.x,
                    object_pose.pose.position.y - 0.18,
                    object_pose.pose.position.z,
                ]
            )
            print("Goal position initialized to:", self.robot.goal_pos)
            self._initialized = True

        joint_pos = self.robot.forward(None)

        if joint_pos is None:
            error_msg = "Joint positions are not available. Skipping servo command."
            raise ValueError(error_msg)

        if len(joint_pos) != 6:
            error_msg = f"Expected 6 joint positions, got {len(joint_pos)}!"
            raise ValueError(error_msg)

        cmd = [0.0] * 6

        for i, pos in enumerate(joint_pos):
            target_pos = self._map_joint_angle(pos, i)
            cmd[i] = target_pos

        t_start = self._rtde_c.initPeriod()

        self._rtde_c.servoJ(cmd, 0.1, 0.2, 1.0 / 100.0, 0.2, 300)
        self._control_joint_states_publisher.publish(JointState(position=cmd))

        self.get_logger().info(
            f"TCP Position: {tcp_pose[0]:.3f}, {tcp_pose[1]:.3f}, {tcp_pose[2]:.3f}"
        )

        # self.get_logger().info(f"Sending servo command: {cmd}")

        self._rtde_c.waitPeriod(t_start)

    def _map_joint_angle(self, pos: float, index: int) -> float:
        """
        Map a simulation joint angle (in radians) to the real-world servo angle (in radians)

        Args:
            pos (float): Joint angle from simulation (in radians)
            index (int): Index of the joint

        Returns:
            float: Mapped joint angle withing the servo limits
        """
        L, U, inversed = self.SIM_DOF_ANGLE_LIMITS[index]
        A, B = self.SERVO_ANGLE_LIMITS[index]
        angle_deg = np.rad2deg(float(pos))
        # Check if the simulation angle is within limits
        if not L <= angle_deg <= U:
            self.get_logger().warn(
                f"Simulation joint {index} angle ({angle_deg}) out of range [{L}, {U}]. Clipping."
            )
            angle_deg = np.clip(angle_deg, L, U)
        # Map the angle from the simulation range to the servo range
        mapped = (angle_deg - L) * ((B - A) / (U - L)) + A
        if inversed:
            mapped = (B - A) - (mapped - A) + A
        # Verify the mapped angle is within servo limits
        if not A <= mapped <= B:
            raise Exception(
                f"Mapped joint {index} angle ({mapped}) out of servo range [{A}, {B}]."
            )
        return mapped

    def _reset(self):
        self._rtde_c.moveJ(self.robot.default_pos[:6])
        self._rtde_c.stopJ()

        for _ in range(10):
            self._control_joint_states_publisher.publish(
                JointState(position=self.robot.default_pos[:6])
            )

        time.sleep(1)

    def stop(self):
        self._rtde_c.stopJ()
        time.sleep(1)


def main(args=None):
    rclpy.init(args=args)

    node = ReachPolicy()

    import threading

    thread1 = threading.Thread(target=rclpy.spin, args=(node,))

    thread1.start()

    rate = node.create_rate(100.0)  # 100 Hz

    while rclpy.ok():
        try:
            node.run()
            rate.sleep()

        except ValueError:
            node.get_logger().warn("ValueError encountered. Skipping iteration.")
            continue

        except KeyboardInterrupt:
            node.get_logger().info("Keyboard interrupt received. Stopping...")
            break
        except Exception as e:
            node.get_logger().error(f"Error in run loop: {e}")
            break

    node.stop()

    node.destroy_node()
    rclpy.shutdown()

    thread1.join()


if __name__ == "__main__":
    main()
