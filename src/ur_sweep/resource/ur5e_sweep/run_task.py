# ROS2
import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.time import Time
# from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default
import tf2_ros

# Message
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

from ur5e_sweep.ur import URReachPolicy



# UR3
import rtde_control
import rtde_receive

# TF
from tf2_ros import *

# Python
import numpy as np
import torch
import math
import time

# UR3
import rtde_control
import rtde_receive

#utils
from utils_pkg.rot_utils import QuaternionAngle, ForwardKinematics

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
    PI = math.pi
    SERVO_ANGLE_LIMITS = [
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
    ]
    
    # ROS topics and joint names
    STATE_TOPIC = '/scaled_joint_trajectory_controller/state'
    CMD_TOPIC = '/scaled_joint_trajectory_controller/joint_trajectory'
    JOINT_NAMES = [
        'elbow_joint',
        'shoulder_lift_joint',
        'shoulder_pan_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]
    
    # Mapping from joint name to simulation action index
    JOINT_NAME_TO_IDX = {
        'elbow_joint': 2,
        'shoulder_lift_joint': 1,
        'shoulder_pan_joint': 0,
        'wrist_1_joint': 3,
        'wrist_2_joint': 4,
        'wrist_3_joint': 5
    }
    
    def __init__(self, fail_quietly: bool = False, verbose: bool = False):
        """Initialize the ReachPolicy node"""
        super().__init__('reach_policy_node')
        
        self.robot = URReachPolicy()
        
        # UR3
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.2.2")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.2.2")
        
        self.target_command = np.zeros(7)
        self.step_size = 1.0/100.0 # 10 ms period = 100 Hz
        
        self.i = 0
        self.fail_quietly = fail_quietly
        self.verbose = verbose
        self.pub_freq = 100.0 # Hz
        self.current_pos = None # Dictionary of current joint positions
        self.target_pos = None
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        self.create_subscription(msg_type=Float32MultiArray, 
                                 topic="/kalman_filter_node/observation", 
                                 callback=self.get_target_pos_in_base_link,
                                 qos_profile=qos_profile_system_default)
        
        self.target_pos = np.zeros(3)
        
        
        # 여기에 타이머 추가
        # self.pose_command_tf_timer = self.create_timer(
        #     self.step_size,  # 10Hz 주기
        #     self.publish_pose_command_tf
        # )

        self.min_traj_dur = 0 # Minimum trajectory duration in seconds
        
        self.timer = self.create_timer(self.step_size, self.step_callback)
        
        self.reset()
        
        self.get_logger().info("ReachPolicy node initialized.")
        

    def map_joint_angle(self, pos: float, index: int) -> float:
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
    
    def get_target_pos_in_base_link(self, msg:Float32MultiArray):
        self.target_pos = np.array(msg.data)
        self.robot.update_target_state(pos=self.target_pos)
        
        
    
    def get_tcp_pose_in_base_link(self):
        try:
            now = self.get_clock().now().to_msg()
            # base_link → tcp 변환 획득
            transform = self.tf_buffer.lookup_transform(
                target_frame="base_link",
                source_frame="tcp",
                time=rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )

            # 위치 텐서
            position_array = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ], dtype=np.float32)

            # 쿼터니언 텐서 (w, x, y, z 순서)
            orientation_array = np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ], dtype=np.float32)
            # 최종 Pose 텐서
            pose_tensor = np.concatenate([position_array, orientation_array])
            # print(position_array)
            self.robot.update_tcp_state(pose=pose_tensor)
        
        except Exception as e:
            self.get_logger().error(f"[TF Transform Error] {e}")
            
        
    def step_callback(self):
        """
        Timer callback to compute and publish the next joint trajectory command.
        """
    
        # Set a constant target command for the robot (example values)
        self.current_pos = self.rtde_r.getActualQ()
        self.current_vel = self.rtde_r.getActualQd()
        # print(self.current_pos)
        # print(self.current_vel)
        self.robot.update_joint_state(self.current_pos, self.current_vel)
        moving_average = 1.0
        self.get_tcp_pose_in_base_link()
        
        if not np.array_equal(self.robot.current_tcp_pose, np.zeros(7)):
            joint_pos = self.robot.forward(self.step_size)
            if joint_pos is not None:
                if len(joint_pos) != 6:
                    raise Exception(f"Expected 6 joint positions, got {len(joint_pos)}!")
                
                cmd = [0] * 6
                
                for i, pos in enumerate(joint_pos):
                    target_pos = self.map_joint_angle(pos, i)
                    # cmd[i] = self.current_pos[i] * (1 - moving_average) + target_pos * moving_average
                    cmd[i] = target_pos
                if self.current_pos is None or cmd is None:
                    return
                

                # time start period
                t_start = self.rtde_c.initPeriod()
                self.rtde_c.servoJ(cmd, 0.1, 0.2, 1.0/100.0, 0.2, 300)
                # self.rtde_c.moveJ(cmd)
                self.rtde_c.waitPeriod(t_start)
            #     # self.get_logger().info(f"current: {self.current_pos}")
            #     # self.get_logger().info(f"target: {joint_pos}")
            self.i += 1
        
    def publish_pose_command_tf(self):
        if self.target_command is None:
            return

        
        pos = self.target_command[:3]
        quat = self.target_command[3:]  # [w, x, y, z]

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "command"

        t.transform.translation.x = pos[0].item()
        t.transform.translation.y = pos[1].item()
        t.transform.translation.z = pos[2].item()

        t.transform.rotation.w = quat[0].item()
        t.transform.rotation.x = quat[1].item()
        t.transform.rotation.y = quat[2].item()
        t.transform.rotation.z = quat[3].item()

        self.tf_broadcaster.sendTransform(t)
        
    def reset(self):
        self.rtde_c.moveJ(self.robot.default_pos[:6] )
        self.rtde_c.stopJ()
        time.sleep(1)
        
    def stop(self):
        self.rtde_c.stopJ()
        time.sleep(1)
def main(args = None):
    rclpy.init(args=args)
    node = ReachPolicy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.robot.shutdown()
        node.stop()
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
        