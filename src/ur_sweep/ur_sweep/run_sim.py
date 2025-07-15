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
import struct

import sys
import os
import socket
import struct

from omni.isaac.kit import SimulationApp


from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from pxr import UsdPhysics, Usd, UsdGeom


class EEFManager(object):
    def __init__(self, node: Node):
        self._node = node

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

    @staticmethod
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

        self._prim_paths = [
            "/World/ur5e/base_link_inertia/shoulder_pan_joint",
            "/World/ur5e/shoulder_link/shoulder_lift_joint",
            "/World/ur5e/upper_arm_link/elbow_joint",
            "/World/ur5e/forearm_link/wrist_1_joint",
            "/World/ur5e/wrist_1_link/wrist_2_joint",
            "/World/ur5e/wrist_2_link/wrist_3_joint",
        ]
        self._prims = [XFormPrim(path) for path in self._prim_paths]

        self._stiffness_attrs = [
            prim.prim.GetAttribute("drive:angular:physics:stiffness")
            for prim in self._prims
        ]
        self._damping_attrs = [
            prim.prim.GetAttribute("drive:angular:physics:damping")
            for prim in self._prims
        ]

        self._target_positions_attrs = [
            prim.prim.GetAttribute("drive:angular:physics:targetPosition")
            for prim in self._prims
        ]
        self._currnet_positions_attrs = [
            prim.prim.GetAttribute("state:angular:physics:position")
            for prim in self._prims
        ]
        self._current_velocities_attrs = [
            prim.prim.GetAttribute("state:angular:physics:velocity")
            for prim in self._prims
        ]

        self._initialized = False
        self._eef_manager = EEFManager(node=self)

        self._reset()

    def run(self):
        current_positions = [attr.Get() for attr in self._currnet_positions_attrs]
        current_velocities = [attr.Get() for attr in self._current_velocities_attrs]

        # Get the current joint states
        joint_states = JointState(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id="base_link",
            ),
            position=[
                current_positions[1],
                current_positions[2],
                current_positions[3],
                current_positions[4],
                current_positions[5],
                current_positions[0],
            ],
            velocity=[
                current_velocities[1],
                current_velocities[2],
                current_velocities[3],
                current_velocities[4],
                current_velocities[5],
                current_velocities[0],
            ],
            effort=[0.0] * 6,  # Effort is not used in this example
        )

        tcp_pose = self._eef_manager.forward_kinematics(joint_states=joint_states)
        if tcp_pose is None:
            error_msg = "TCP pose is not available. Skipping servo command."
            raise ValueError(error_msg)

        # Set the stiffness and damping for the joints
        stiffness_values = [600.0, 1000.0, 1000.0, 1000.0, 600.0, 600.0]
        damping_values = [40.0, 100.0, 100.0, 100.0, 70.0, 70.0]
        for attr, stiff, damp in zip(
            self._stiffness_attrs, stiffness_values, self._damping_attrs, damping_values
        ):
            attr.Set(stiff)
            damp.Set(damp)

        # STEP 1: Get the current joint positions and velocities
        joint_states: JointState = EEFManager.reorder_joint_states(joint_states)

        current_pos = joint_states.position
        current_vel = joint_states.velocity

        # STEP 2: Get the current TCP pose in base_link frame
        # Update the robot's joint state
        self.robot.update_joint_state(current_pos, current_vel)
        # Update the robot's TCP state

        self.robot.update_tcp_state(pose=tcp_pose)  # tcp_pose

        self.robot.update_target_state(pos=[0.722, -0.153, 0.27])
        self.robot.update_goal_state(goal=[0.722, -0.153 - 0.18, 0.27])

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

        # Set Target Positions
        for i, target_pos in enumerate(cmd):
            self._target_positions_attrs[i].Set(target_pos)

        self.get_logger().info(
            f"TCP Position: {tcp_pose[0]:.3f}, {tcp_pose[1]:.3f}, {tcp_pose[2]:.3f}"
        )

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
        self.get_logger().info("Resetting robot to default position...")

        default_joint = self.robot.default_pos[:6].tolist()
        for _ in range(10):
            for i, target_pos in enumerate(default_joint):
                self._target_positions_attrs[i].Set(target_pos)

        time.sleep(1)

    def stop(self):
        time.sleep(1)


def main(args=None):
    rclpy.init(args=args)

    # Start Isaac Sim app
    simulation_app = SimulationApp(
        {"headless": False}
    )  # Set headless=True to run without GUI

    node = ReachPolicy()

    import threading

    thread1 = threading.Thread(target=rclpy.spin, args=(node,))

    thread1.start()

    rate = node.create_rate(100.0)  # 100 Hz

    while rclpy.ok():
        try:
            node.run()
            rate.sleep()

        except ValueError as ve:
            node.get_logger().error(f"ValueError encountered: {ve}")
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
