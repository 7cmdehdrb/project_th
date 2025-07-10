import os
import sys
import numpy as np
import torch
from typing import List, Tuple, Optional, Any
from ur_sweep.policy_controller import PolicyController
from rclpy.node import Node


class URReachPolicy(PolicyController):
    """Policy controller for UR Reach using a pre-trained policy model"""

    def __init__(self, node: Node, model_file: str, yaml_file: str) -> None:
        """Initialize the URReachPolicy instance."""
        super().__init__()

        assert os.path.exists(model_file), f"Model file {model_file} does not exist."
        assert os.path.exists(yaml_file), f"YAML file {yaml_file} does not exist."

        self._node: Node = node

        self.dof_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        self.load_policy(
            model_file,
            yaml_file,
        )

        # >>> Flag Variables >>>
        self.policy_counter = 0
        self.has_joint_data: bool = False
        self.has_tcp_data: bool = False

        # >>> Observation Vectors >>>
        self.action_scale = 0.5
        self.previous_action = np.zeros(
            7
        )  # Manipulator 6 + gripper 1 (gripper open/close)
        self.action = np.zeros(7)  # Manipulator 6 + gripper 1 (gripper open/close)
        self.current_joint_positions = np.zeros(8, dtype=np.float32)
        self.current_joint_velocities = np.zeros(8, dtype=np.float32)
        self.current_tcp_pose = np.zeros(7, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.goal_pos = np.zeros(3, dtype=np.float32)

    def update_joint_state(self, position: List[float], velocity: List[float]) -> None:
        """
        Update the current joint state.
        Args:
            position (_type_): A list or array of joint positions
            velocity (_type_): A list or array of joint velocities
        """
        self.current_joint_positions = np.array(
            position[: len(self.dof_names)], dtype=np.float32
        )
        self.current_joint_velocities = np.array(
            velocity[: len(self.dof_names)], dtype=np.float32
        )

        self.has_joint_data = True

    def update_tcp_state(self, pose: List[float]) -> None:
        """
        Update the current tcp state.

        Args:
            pose (_type_): A list or array of tcp point
        """
        self.current_tcp_pose = np.array(pose)

        self.has_tcp_data = True

    def update_target_state(self, pos: np.ndarray) -> None:
        self.target_pos = pos

    def update_goal_state(self, goal: np.ndarray) -> None:
        self.goal_pos = goal

    def compute_observation(self) -> np.ndarray:
        """
        Compute the observation vector for the policy network.

        Args:
            command (np.ndarray): The target command vector

        Returns:
            np.ndarray: An observation vector if joint data is available, otherwise None.
        """

        if not self.has_joint_data or not self.has_tcp_data:
            self._node.get_logger().warn(
                "Joint positions, velocities, or TCP pose are not initialized properly."
            )
            return None

        obs = np.zeros(35)
        obs[:6] = self.current_joint_positions - self.default_pos[:6]

        if self.previous_action[-1] < 0:
            obs[6:8] = 0.4
        else:
            obs[6:8] = 0.04

        obs[8:14] = self.current_joint_velocities
        obs[14:21] = self.previous_action
        obs[21:24] = self.target_pos
        obs[24] = 0.07
        obs[25:32] = self.current_tcp_pose
        obs[32:35] = self.goal_pos

        obs = np.expand_dims(obs, axis=0).astype(np.float32)

        # print(f"Observation: {obs}")

        return obs

    def forward(self, dt: float) -> np.ndarray:
        """
        Compute the next joint positions based on the policy

        Args:
            dt (float): Time step for the forward pass.
            command (np.ndarray): The target command vector.

        Returns:
            np.ndarray: The computed joint positions if joint data is available.
        """
        if np.array_equal(self.current_tcp_pose, np.zeros(7)):
            self._node.get_logger().warn(
                "TCP pose is not initialized properly. Skipping forward pass."
            )
            return None

        if not self.has_joint_data or not self.has_tcp_data:
            self._node.get_logger().warn(
                "Joint positions, velocities, or TCP pose are not initialized properly."
            )
            return None

        obs = None

        if self.policy_counter % self._decimation == 0:
            obs = self.compute_observation()
            if obs is None:
                return None
            self.previous_action = self.action.copy()
            self.action = self._compute_action(obs)

        processed_action = self.action * self.action_scale
        joint_positions = processed_action[:6] + self.default_pos[:6]

        self.policy_counter += 1
        return joint_positions
