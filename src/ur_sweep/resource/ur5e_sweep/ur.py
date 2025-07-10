import numpy as np
import torch
from ur5e_sweep.policy_controller import PolicyController
import os
import sys


class URReachPolicy(PolicyController):
    """Policy controller for UR Reach using a pre-trained policy model"""
    
    def __init__(self) -> None:
        """Initialize the URReachPolicy instance."""
        
        super().__init__()
        
        self.dof_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        # Load the pre-trained policy model and environment configuration
        # YOU NEED TO  CHANGE THE PATH
        
        self.load_policy(
            "/home/irol2/ros2_ws/src/ur5e_sweep/resource/7_set/exported/policy.pt",
            "/home/irol2/ros2_ws/src/ur5e_sweep/resource/7_set/params/env.yaml",
        )
        
        self._action_scale = 0.5
        self._previous_action = np.zeros(7)
        self.action = np.zeros(7)
        self._policy_counter = 0
        self.has_joint_data = False
        self.has_tcp_data = False
        self.current_joint_positions = np.zeros(8, dtype=np.float32)
        self.current_joint_velocities = np.zeros(8, dtype=np.float32)
        self.current_tcp_pose = np.zeros(7, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)
        
        
        # Data logs
        self._csv_file = open("/home/irol2/ros2_ws/data/log01.csv", mode="w", encoding="utf-8")
        header_joint = [f"joint_{i}" for i in range(8)]
        header_joint_vel = [f"joint_vel_{i}" for i in range(6)]
        header_prev_action = [f"prev_a_{i}" for i in range(7)]
        header_target_obs_pos = [f"target_obs_pos_{i}" for i in range(3)]
        header_target_width = ["target_width"]
        header_current_tcp = [f"current_tcp_{i}" for i in range(7)]
        header_goal_pos = [f"goal_pos_{i}" for i in range(3)]
        header_action = [f"action_{i}" for i in range(7)]
        header = header_joint + header_joint_vel + header_prev_action + header_target_obs_pos + header_target_width +header_current_tcp + header_goal_pos + header_action

        header_txt = ""
        for i, h in enumerate(header):
            header_txt += h
            header_txt += "," if i != (len(header) - 1) else "\n"
            
        self._csv_file.write(header_txt)
        
    def update_joint_state(self, position, velocity) -> None:
        """
        Update the current joint state.
        Args:
            position (_type_): A list or array of joint positions
            velocity (_type_): A list or array of joint velocities
        """
        
        self.current_joint_positions = np.array(position[:self.num_joints], dtype=np.float32)
        self.current_joint_velocities = np.array(velocity[:self.num_joints], dtype=np.float32)
        # print(f"joint: {self.current_joint_positions}")
        self.has_joint_data = True
        
    def update_tcp_state(self, pose) -> None:
        """
        Update the current tcp state.

        Args:
            pose (_type_): A list or array of tcp point
        """
        self.current_tcp_pose = np.array(pose)
        self.has_tcp_data = True
        
    def update_target_state(self, pos) -> None:
        self.target_pos = pos
        
        
        
    def _compute_observation(self) -> np.ndarray:
        """
        Compute the observation vector for the policy network.

        Args:
            command (np.ndarray): The target command vector

        Returns:
            np.ndarray: An observation vector if joint data is available, otherwise None.
        """
        
        if not (self.has_joint_data or self.has_tcp_data):
            return None
        
        obs = np.zeros(35)
        obs[:6] = self.current_joint_positions - self.default_pos[:6]
        
        if self._previous_action[-1] < 0:
            obs[6:8] = 0.4
        else:
            obs[6:8] = 0.04
            
        obs[8:14] = self.current_joint_velocities
        obs[14:21] = self._previous_action
        obs[21:24] = [0.75, 0.2 ,0.2549] #self._target_object_pos
        obs[24] = 0.07 #self._target_object_width 
        obs[25:32] = self.current_tcp_pose
        obs[32:35] = [0.75, 0.02 ,0.25495] #self._goal_pos
        
        obs= np.expand_dims(obs, axis=0).astype(np.float32)
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
        
        if not (self.has_joint_data or self.has_tcp_data):
            return None
        
        obs = None

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation()
            if obs is None:
                return None
            self._previous_action = self.action.copy()
            self.action = self._compute_action(obs)

        processed_action =  self.action * self._action_scale
        joint_positions = processed_action[:6] + self.default_pos[:6]
        # preint(f"dq: {joint_positions - self.current_joint_positions}")
        print(joint_positions)

        self._policy_counter += 1
        
        if obs is not None:    
            
            obs_list = obs.tolist() # 35
            action_list = self.action.tolist() # 7
            
            observation_txt = ""
            for o in obs_list:
                observation_txt += str(o)
                observation_txt += ","
                
            for i, a in enumerate(action_list):
                observation_txt += str(a)
                observation_txt += "," if i != len(action_list) - 1 else "\n"
            
            observation_txt = observation_txt.replace("[","").replace("]","")
              
            self._csv_file.write(observation_txt)
        
        return joint_positions
    

    def shutdown(self):
        self._csv_file.close()