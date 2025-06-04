# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default
from builtin_interfaces.msg import Duration as BuiltinDuration

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from moveit_msgs.msg import *
from moveit_msgs.srv import *
from trajectory_msgs.msg import *

# TF
from tf2_ros import *

# Python
import numpy as np
from enum import Enum
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

# ROS2 Custom Package
from moveit2_commander import (
    FK_ServiceManager,
    IK_ServiceManager,
    GetPlanningScene_ServiceManager,
    ApplyPlanningScene_ServiceManager,
    CartesianPath_ServiceManager,
    KinematicPath_ServiceManager,
    ExecuteTrajectory_ServiceManager,
)
from path_planner.potential_field_path_planner import (
    PotentialPoint,
    PotentialObstacle,
    PotentialFieldPlanner,
)


class Status(Enum):
    WAITING = 0
    PLANNING = 1
    EXECUTING = 2
    HOMING = 3


class JointStateManager:
    def __init__(self, node: Node):
        self._node = node
        self._joint_states: JointState = None

        # Subscriber to joint states
        self._joint_state_subscriber = self._node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            qos_profile_system_default,
        )

    def _joint_state_callback(self, msg: JointState):
        """
        Callback function to update the joint states.
        """
        self._joint_states = msg

    @property
    def joint_states(self) -> JointState:
        """
        Returns the latest joint states.
        """
        if self._joint_states is None:
            self._node.get_logger().warn("Joint states not yet received.")
            return None

        return self._joint_states


def parse_obstacles_to_marker_array(
    obstacles: List[PotentialObstacle], z: float, header: Header
) -> MarkerArray:
    marker_array = MarkerArray()

    for i, obstacle in enumerate(obstacles):
        obstacle: PotentialObstacle

        marker = Marker()
        marker.header = header
        marker.id = i
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = obstacle.x
        marker.pose.position.y = obstacle.y
        marker.pose.position.z = 0.25 + 0.05
        marker.scale.x = obstacle.r
        marker.scale.y = obstacle.r
        marker.scale.z = 0.12

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.3  # Fully opaque

        marker_array.markers.append(marker)

    return marker_array


def parse_np_path_to_path(path: np.ndarray, z: float, header: Header) -> Path:
    """
    Converts a numpy array path to a ROS Path message.
    """
    ros_path = Path()
    ros_path.header = header

    for point in path:
        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = z  # Assuming a flat plane
        ros_path.poses.append(pose)

    return ros_path


def parse_np_path_to_pose_array(
    path: np.ndarray, z: float, orientation: Quaternion
) -> PoseArray:
    """
    Converts a numpy array path to a ROS PoseArray message.
    """
    pose_array = []

    for point in path:
        pose = Pose()
        pose.position.x = point[0]
        pose.position.y = point[1]
        pose.position.z = z  # Assuming a flat plane
        pose.orientation = orientation  # Use a fixed orientation

        pose_array.append(pose)

    return pose_array


class MainNode(Node):
    def __init__(self):
        super().__init__("robot_control_node")

        self._status = Status.WAITING
        self._methods = {
            Status.WAITING: self._waiting,
            Status.PLANNING: self._planning,
            Status.EXECUTING: self._executing,
            Status.HOMING: self._homing,
        }

        # Initialize the potential field planner
        self._pf_planner = PotentialFieldPlanner(
            rr=0.05,  # robot radius [m]
            resolution=0.02,  # grid resolution [m]
            kp=1.0,  # attractive potential gain
            eta=100.0,  # repulsive potential gain
            area_offset=1.0,  # potential area width [m]
            oscillations=3,  # number of previous positions to check for oscillations
        )

        # >>> MoveIt2 Service Managers
        self._fk_manager = FK_ServiceManager(self)
        self._ik_manager = IK_ServiceManager(self)
        self._get_planning_scene_manager = GetPlanningScene_ServiceManager(self)
        self._apply_planning_scene_manager = ApplyPlanningScene_ServiceManager(self)
        self._cartesian_path_manager = CartesianPath_ServiceManager(
            self, planning_group="ur_manipulator", fraction_threshold=0.8
        )
        self._kinematic_path_manager = KinematicPath_ServiceManager(
            self, planning_group="ur_manipulator"
        )
        self._execute_trajectory_manager = ExecuteTrajectory_ServiceManager(self)
        # <<< MoveIt2 Service Managers

        # >>> Parameters >>>
        self._fk_pose: PoseStamped = None
        self._pfp_path: np.ndarray = None
        # <<< Parameters <<<

        self._joint_state_manager = JointStateManager(self)

        # Create a publisher
        self._marker_publisher = self.create_publisher(
            MarkerArray, "/obstacle_markers", qos_profile=qos_profile_system_default
        )
        self._path_publisher = self.create_publisher(
            Path, "/planned_path", qos_profile=qos_profile_system_default
        )

        # self._timer = self.create_timer(0.5, self.run)

    def run(self):
        self._methods[self._status]()

    def _update_status(self):
        if self._status == Status.HOMING:
            self._status = Status.WAITING
            self.get_logger().info("Homing completed. Status set to WAITING.")

        else:
            self._status = Status(self._status.value + 1)

        return self._status.value

    # >>> State Machine Methods
    def _waiting(self):
        """
        1. Get FK Position
        """
        if self._joint_state_manager.joint_states is None:
            self.get_logger().warn("Joint states not available yet.")
            return None

        try:
            print("Calculating FK Pose...")

            self._fk_pose: PoseStamped = self._fk_manager.run(
                joint_states=self._joint_state_manager.joint_states,
                end_effector="wrist_3_link",
            )

            self.get_logger().info(
                f"FK Pose: {self._fk_pose.pose.position.x:.3f}, {self._fk_pose.pose.position.y:.3f}, {self._fk_pose.pose.position.z:.3f}"
            )

            self._update_status()

        except Exception as e:
            self.get_logger().error(f"FK calculation failed: {e}")
            return None

    def _planning(self):
        """
        2. Plan Path
        - Start: fk_pose
        - Goal: user-defined goal pose
        - Obstacles: user-defined obstacles
        """

        try:
            start_point = PotentialPoint(
                x=self._fk_pose.pose.position.x,
                y=self._fk_pose.pose.position.y,
            )
            goal_point = PotentialPoint(x=-0.3, y=0.6)  # Example goal point

            obstacles = [
                PotentialObstacle(x=0.0, y=0.75, r=0.05),
                PotentialObstacle(x=0.2, y=0.75, r=0.05),
                PotentialObstacle(x=-0.2, y=0.75, r=0.05),
                PotentialObstacle(x=0.0, y=0.6, r=0.05),
                PotentialObstacle(x=0.2, y=0.6, r=0.05),
                PotentialObstacle(x=-0.2, y=0.6, r=0.05),
            ]

            self._pfp_path = self._pf_planner.planning(
                start=start_point,
                goal=goal_point,
                obstacles=obstacles,
            )

            self._marker_publisher.publish(
                parse_obstacles_to_marker_array(
                    obstacles,
                    self._fk_pose.pose.position.z,
                    Header(frame_id="world", stamp=self.get_clock().now().to_msg()),
                )
            )
            self._path_publisher.publish(
                parse_np_path_to_path(
                    self._pfp_path,
                    self._fk_pose.pose.position.z,
                    Header(frame_id="world", stamp=self.get_clock().now().to_msg()),
                )
            )

            print("Planning completed.")

            self._update_status()

        except Exception as e:
            self.get_logger().error(f"Planning failed: {e}")
            return None

    def _executing(self):
        """
        3. Execute Path (Cartesian Path)
        """
        try:
            sliced_path = np.vstack(
                [self._pfp_path[::50], self._pfp_path[-1]]
            )  # Slicing the path for execution

            traj: RobotTrajectory = self._cartesian_path_manager.run(
                header=Header(frame_id="world", stamp=self.get_clock().now().to_msg()),
                waypoints=parse_np_path_to_pose_array(
                    sliced_path,
                    self._fk_pose.pose.position.z,
                    self._fk_pose.pose.orientation,
                ),
                end_effector="wrist_3_link",
                joint_states=self._joint_state_manager.joint_states,
            )

            last_traj: JointTrajectoryPoint = traj.joint_trajectory.points[-1]

            execution_time = last_traj.time_from_start
            execution_time_float = execution_time.sec + execution_time.nanosec * 1e-9

            target_execution_time = 5.0
            executution_time_ratio = execution_time_float / target_execution_time

            if traj is not None:
                scaled_traj = self._execute_trajectory_manager.scale_trajectory(
                    trajectory=traj,
                    scale_factor=executution_time_ratio,
                )

                self._execute_trajectory_manager.run(
                    trajectory=scaled_traj,
                )

                self._update_status()

        except Exception as e:
            self.get_logger().error(f"Execution failed: {e}")
            return None

    def _homing(self):
        try:
            home_joint_states = self._ik_manager.run(
                pose_stamped=self._fk_pose,
                joint_states=self._joint_state_manager.joint_states,
                end_effector="wrist_3_link",
            )

            if home_joint_states is not None:
                path: RobotTrajectory = self._cartesian_path_manager.run(
                    header=Header(
                        frame_id="world", stamp=self.get_clock().now().to_msg()
                    ),
                    waypoints=[self._fk_pose.pose],
                    joint_states=self._joint_state_manager.joint_states,
                    end_effector="wrist_3_link",
                )

                if path is not None:
                    self._execute_trajectory_manager.run(
                        trajectory=path,
                    )

                    self.get_logger().info("Homing completed successfully.")
                    self._update_status()

        except Exception as e:
            self.get_logger().error(f"Homing failed: {e}")
            return None

    # <<< State Machine Methods


def main():
    rclpy.init(args=None)

    node = MainNode()

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 1.0  # Frequency in Hz
    rate = node.create_rate(hz)

    try:
        while rclpy.ok():
            node.run()
            rate.sleep()

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
