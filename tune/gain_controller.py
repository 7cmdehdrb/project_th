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

from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


def merge_robot_trajectories(
    rt1: RobotTrajectory, rt2: RobotTrajectory
) -> RobotTrajectory:
    """
    두 개의 RobotTrajectory를 받아 joint_trajectory를 시간 기준으로 연결한 새로운 RobotTrajectory를 반환한다.
    time_from_start는 연속되도록 자동 조정된다.
    """
    jt1: JointTrajectory = rt1.joint_trajectory
    jt2: JointTrajectory = rt2.joint_trajectory

    if jt1.joint_names != jt2.joint_names:
        raise ValueError("Joint names of both trajectories must match.")

    merged_rt = RobotTrajectory()
    merged_jt = JointTrajectory()
    merged_jt.joint_names = jt1.joint_names
    merged_jt.header = jt1.header  # 첫 번째 trajectory의 header 유지

    # 첫 번째 트래젝토리 포인트 복사
    merged_jt.points = jt1.points.copy()

    # 마지막 시점 (초 + 나노초) 계산
    if jt1.points:
        last_time = jt1.points[-1].time_from_start
    else:
        last_time = Duration(sec=0, nanosec=0)

    base_nsec = last_time.sec * 1_000_000_000 + last_time.nanosec

    # 두 번째 trajectory의 포인트를 시간 보정 후 추가
    for point in jt2.points:
        new_point = JointTrajectoryPoint()
        new_point.positions = point.positions
        new_point.velocities = point.velocities
        new_point.accelerations = point.accelerations
        new_point.effort = point.effort

        # 기존 시간 + base offset
        point_nsec = (
            point.time_from_start.sec * 1_000_000_000 + point.time_from_start.nanosec
        )
        total_nsec = base_nsec + point_nsec

        new_point.time_from_start.sec = total_nsec // 1_000_000_000
        new_point.time_from_start.nanosec = total_nsec % 1_000_000_000

        merged_jt.points.append(new_point)

    merged_rt.joint_trajectory = merged_jt
    return merged_rt


class GainTuner(Node):
    def __init__(self):
        super().__init__("gain_tuner")

        self._params = {
            "joint_param/shoulder_pan/damping": self.create_publisher(
                Float32,
                "joint_param/shoulder_pan/damping",
                qos_profile=qos_profile_system_default,
            ),
            "joint_param/shoulder_pan/stiffness": self.create_publisher(
                Float32,
                "joint_param/shoulder_pan/stiffness",
                qos_profile=qos_profile_system_default,
            ),
            "joint_param/shoulder_lift/damping": self.create_publisher(
                Float32,
                "joint_param/shoulder_lift/damping",
                qos_profile=qos_profile_system_default,
            ),
            "joint_param/shoulder_lift/stiffness": self.create_publisher(
                Float32,
                "joint_param/shoulder_lift/stiffness",
                qos_profile=qos_profile_system_default,
            ),
            "joint_param/elbow/damping": self.create_publisher(
                Float32,
                "joint_param/elbow/damping",
                qos_profile=qos_profile_system_default,
            ),
            "joint_param/elbow/stiffness": self.create_publisher(
                Float32,
                "joint_param/elbow/stiffness",
                qos_profile=qos_profile_system_default,
            ),
            "joint_param/wrist1/damping": self.create_publisher(
                Float32,
                "joint_param/wrist1/damping",
                qos_profile=qos_profile_system_default,
            ),
            "joint_param/wrist1/stiffness": self.create_publisher(
                Float32,
                "joint_param/wrist1/stiffness",
                qos_profile=qos_profile_system_default,
            ),
            "joint_param/wrist2/damping": self.create_publisher(
                Float32,
                "joint_param/wrist2/damping",
                qos_profile=qos_profile_system_default,
            ),
            "joint_param/wrist2/stiffness": self.create_publisher(
                Float32,
                "joint_param/wrist2/stiffness",
                qos_profile=qos_profile_system_default,
            ),
            "joint_param/wrist3/damping": self.create_publisher(
                Float32,
                "joint_param/wrist3/damping",
                qos_profile=qos_profile_system_default,
            ),
            "joint_param/wrist3/stiffness": self.create_publisher(
                Float32,
                "joint_param/wrist3/stiffness",
                qos_profile=qos_profile_system_default,
            ),
        }

        self._joint_state_sub = SimpleSubscriberManager(
            self, "joint_states", JointState, qos_profile=qos_profile_system_default
        )

        # >>> MoveIt2 Service Managers >>>
        self._fk_service_manager = FK_ServiceManager(self)
        self._ik_service_manager = IK_ServiceManager(self)
        self._kinematic_path_service_manager = KinematicPath_ServiceManager(self)
        self._cartesian_path_service_manager = CartesianPath_ServiceManager(self)
        self._execute_trajectory_service_manager = ExecuteTrajectory_ServiceManager(
            self
        )
        # <<< MoveIt2 Service Managers <<<

        self._home_joints = JointState(
            name=[
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "shoulder_pan_joint",
            ],
            position=[
                -2.2,
                2.2,
                0.0,
                1.57,
                np.pi / 2.0,
                0.0,
            ],
        )
        self._goal_joints = JointState(
            name=[
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "shoulder_pan_joint",
            ],
            position=[
                -1.2712636640497141,
                1.2026881583600584,
                0.049155477843335404,
                1.5441581584117694,
                1.2780465746091123,
                0.3549688634351176,
            ],
        )
        self._exp_num_pub = self.create_publisher(
            UInt32, "experiment_number", qos_profile=qos_profile_system_default
        )

        self._dampings = np.linspace(1.0, 100.0, num=10)  # 0~100
        self._stiffnesses = np.linspace(10.0, 100.0, num=10)  # 200~1400
        self._exp_num = 0

    def run(self):
        # for _ in range():  # Run 10 iterations for each combination
        self._exp_num = 0
        for self._damping in self._dampings:
            for self._stiffness in self._stiffnesses:

                # if self._exp_num < 776:
                #     self.get_logger().info(
                #         f"Skipping experiment {self._exp_num} with damping={self._damping}, stiffness={self._stiffness}"
                #     )
                #     self._exp_num += 1
                #     continue

                self._homing()

                while True:
                    traj: RobotTrajectory = self._get_random_trajectory()
                    if traj is not None:
                        break

                param = {
                    "joint_param/shoulder_pan/damping": self._damping,
                    "joint_param/shoulder_pan/stiffness": self._stiffness,
                    "joint_param/shoulder_lift/damping": self._damping,
                    "joint_param/shoulder_lift/stiffness": self._stiffness,
                    "joint_param/elbow/damping": self._damping,
                    "joint_param/elbow/stiffness": self._stiffness,
                    "joint_param/wrist1/damping": self._damping,
                    "joint_param/wrist1/stiffness": self._stiffness,
                    "joint_param/wrist2/damping": self._damping,
                    "joint_param/wrist2/stiffness": self._stiffness,
                    "joint_param/wrist3/damping": self._damping,
                    "joint_param/wrist3/stiffness": self._stiffness,
                }

                # self.get_logger().info(f"Publishing parameters: {param}")

                for _ in range(30):
                    self._publish_param(param)
                    self._exp_num_pub.publish(UInt32(data=self._exp_num))

                self._execute_trajectory_service_manager.run(
                    trajectory=traj,
                )

                self._exp_num += 1

    def _publish_param(self, param: dict):
        for key, value in param.items():
            if key in self._params:
                msg = Float32(data=value)
                self._params[key].publish(msg)
            else:
                self.get_logger().warn(f"Parameter {key} not found in publishers.")

    def _get_random_valid_goal(self) -> Pose:
        min_radius = 0.5
        max_radius = 0.7
        max_angle_deg = 20.0

        # Step 1: Random point on a shell (uniform on sphere)
        vec = np.random.normal(0, 1, size=3)
        vec /= np.linalg.norm(vec)

        # Step 2: Random radius in [min_radius, max_radius]
        radius = np.random.uniform(min_radius, max_radius)
        position = radius * vec

        # Step 3: Direction vector from origin to position (i.e., "outward")
        outward = position / np.linalg.norm(position)

        # Step 4: Generate random direction within `max_angle_deg` from outward
        # Method: sample random axis perpendicular to outward, rotate by random angle <= max_angle_deg
        max_angle_rad = np.radians(max_angle_deg)

        # Generate a random unit vector perpendicular to outward
        rand_vec = np.random.normal(0, 1, 3)
        rand_vec -= rand_vec.dot(outward) * outward
        rand_vec /= np.linalg.norm(rand_vec)

        # Random rotation angle from 0 to max_angle_rad
        theta = np.random.uniform(0, max_angle_rad)
        rot = R.from_rotvec(rand_vec * theta)

        # Apply rotation to outward to get the final direction
        direction = rot.apply(outward)

        # Convert to quaternion: we assume z-axis forward, so align z with `direction`
        # We'll use scipy's Rotation.align_vectors
        result, _ = R.align_vectors([direction], [[0, 0, 1]])
        orientation_quat = result.as_quat()  # [x, y, z, w]

        pose = Pose(
            position=Point(**dict(zip(["x", "y", "z"], position))),
            orientation=Quaternion(**dict(zip(["x", "y", "z", "w"], orientation_quat))),
        )

        print(f"Generated random goal pose: {pose}")

        return pose

    def _get_random_trajectory(self) -> RobotTrajectory:
        if self._joint_state_sub.data is None:
            self.get_logger().warn("Joint states not available yet.")
            return None

        try:
            # 1. Get a random valid goal pose
            goal_pose: Pose = self._get_random_valid_goal()

            # 2. Create path to the goal pose with MoveIt2
            goal_robot_states: RobotState = self._ik_service_manager.run(
                pose_stamped=PoseStamped(
                    header=Header(
                        frame_id="base_link",
                        stamp=self.get_clock().now().to_msg(),
                    ),
                    pose=goal_pose,
                ),
                joint_states=self._joint_state_sub.data,
                end_effector="gripper_link",
                avoid_collisions=False,
            )

            goal_constraints = self._kinematic_path_service_manager.get_goal_constraint(
                goal_joint_states=goal_robot_states.joint_state, tolerance=0.05
            )
            traj: RobotTrajectory = self._kinematic_path_service_manager.run(
                goal_constraints=[goal_constraints],
                path_constraints=None,
                joint_states=self._joint_state_sub.data,
                num_planning_attempts=30,
                allowed_planning_time=1.0,
                max_acceleration_scaling_factor=1.0,
                max_velocity_scaling_factor=1.0,
            )

            if traj is None:
                self.get_logger().warn("No trajectory found for the random goal.")
                return None

            scaled_traj: RobotTrajectory = (
                self._execute_trajectory_service_manager.scale_trajectory(
                    trajectory=traj, scale_factor=0.5
                )
            )

            return scaled_traj

        except Exception as e:
            self.get_logger().error(f"Error executing random goal: {e}")
            return None

    def _homing(self):
        if self._joint_state_sub.data is None:
            self.get_logger().warn("Joint states not available yet.")
            return None

        self.get_logger().info("Homing to the initial position...")

        goal_constraints = self._kinematic_path_service_manager.get_goal_constraint(
            goal_joint_states=self._home_joints, tolerance=0.05
        )
        traj: RobotTrajectory = self._kinematic_path_service_manager.run(
            goal_constraints=[goal_constraints],
            path_constraints=None,
            joint_states=self._joint_state_sub.data,
            num_planning_attempts=30,
            allowed_planning_time=1.0,
            max_acceleration_scaling_factor=1.0,
            max_velocity_scaling_factor=1.0,
        )

        if traj is None:
            self.get_logger().warn("No trajectory found for the random goal.")
            return None

        scaled_traj: RobotTrajectory = (
            self._execute_trajectory_service_manager.scale_trajectory(
                trajectory=traj, scale_factor=0.5
            )
        )

        self._execute_trajectory_service_manager.run(
            trajectory=scaled_traj,
        )


def main():
    rclpy.init(args=None)

    node = GainTuner()

    import threading

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 10.0
    rate = node.create_rate(hz)

    node.run()

    while rclpy.ok():
        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()

    thread.join()


if __name__ == "__main__":
    main()
