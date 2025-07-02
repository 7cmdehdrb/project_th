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
from tf2_msgs.msg import TFMessage
from tf2_geometry_msgs.tf2_geometry_msgs import PoseStamped as TF2PoseStamped
from controller_manager_msgs.srv import SwitchController, ListControllers
from controller_manager_msgs.msg import ControllerState

# TF
from tf2_ros import *

# Python
import numpy as np
from enum import Enum
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from collections import deque
import rotutils

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
from base_package.manager import SimpleSubscriberManager, TransformManager


class SimObjectTransformManager(object):
    # ONLY FOR ISSAC SIM
    def __init__(self, node: Node):
        self._node = node

        self._sim_sub = self._node.create_subscription(
            TFMessage,
            "/isaac_sim/tf/mug",
            self._callback,
            qos_profile_system_default,
        )
        self._pub = self._node.create_publisher(
            PoseStamped,
            "/isaac_sim/pose/mug",
            qos_profile=qos_profile_system_default,
        )

        self._data: PoseStamped = None

    @property
    def data(self) -> PoseStamped:
        """
        Returns the latest pose data.
        """
        if self._data is None:
            self._node.get_logger().warn("Pose data not yet received.")
            return None

        return self._data

    def _publish_data(self, data: PoseStamped):
        """
        Publishes the pose data.
        """
        if data is not None:
            self._pub.publish(data)
        else:
            self._node.get_logger().warn("No data to publish.")

    def _callback(self, msg: TFMessage):
        if len(msg.transforms) == 1:
            transform: TransformStamped = msg.transforms[0]

            data = PoseStamped(
                header=Header(
                    frame_id="world",
                    stamp=self._node.get_clock().now().to_msg(),
                ),
                pose=Pose(
                    position=Point(
                        x=transform.transform.translation.x,
                        y=transform.transform.translation.y,
                        z=transform.transform.translation.z,
                    ),
                    orientation=Quaternion(
                        x=transform.transform.rotation.x,
                        y=transform.transform.rotation.y,
                        z=transform.transform.rotation.z,
                        w=transform.transform.rotation.w,
                    ),
                ),
            )

            if self._data is None:
                self._data = data

            else:
                self._data = data
                self._publish_data(self._data)


class ObjectTransformManager:
    # ONLY FOR REAL ROBOT
    def __init__(self, node: Node):
        self._node = node

        self._transform_manager = TransformManager(node=self._node)

        self._sim_sub = self._node.create_subscription(
            TFMessage,
            "/isaac_sim/tf/mug",
            self._callback,
            qos_profile_system_default,
        )
        self._pub = self._node.create_publisher(
            PoseStamped,
            "/isaac_sim/pose/mug",
            qos_profile=qos_profile_system_default,
        )

        self._data: PoseStamped = None

    @property
    def data(self) -> PoseStamped:
        if self._data is None:
            self._node.get_logger().warn("Pose data not yet received.")
            return None

        tf_pose = TF2PoseStamped(
            header=self._data.header,
            pose=self._data.pose,
        )

        transformed_tf_pose: PoseStamped = self._transform_manager.transform_pose(
            pose=tf_pose,
            target_frame="world",
            source_frame=self._data.header.frame_id,
        )

        return PoseStamped(
            header=transformed_tf_pose.header,
            pose=transformed_tf_pose.pose,
        )

    def _publish_data(self, data: PoseStamped):
        """
        Publishes the pose data.
        """
        if data is not None:
            self._pub.publish(data)
        else:
            self._node.get_logger().warn("No data to publish.")

    def _callback(self, msg: TFMessage):
        if len(msg.transforms) == 1:
            transform: TransformStamped = msg.transforms[0]

            data = PoseStamped(
                header=Header(
                    frame_id="world",
                    stamp=self._node.get_clock().now().to_msg(),
                ),
                pose=Pose(
                    position=Point(
                        x=transform.transform.translation.x,
                        y=transform.transform.translation.y,
                        z=transform.transform.translation.z,
                    ),
                    orientation=Quaternion(
                        x=transform.transform.rotation.x,
                        y=transform.transform.rotation.y,
                        z=transform.transform.rotation.z,
                        w=transform.transform.rotation.w,
                    ),
                ),
            )

            if self._data is None:
                self._data = data

            else:
                self._data = data
                self._publish_data(self._data)


class ControllerSwitcher(object):
    def __init__(self, node: Node):
        self._node = node

        # >>> Initialize /controller_manager/switch_controller >>>
        self._switch_cli = self._node.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        while not self._switch_cli.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info(
                "Waiting for /controller_manager/switch_controller service..."
            )

        self._node.get_logger().info(
            "Connected to /controller_manager/switch_controller service."
        )

        # <<< Initialize /controller_manager/switch_controller <<<

        # >>> Initialize /controller_manager/list_controllers >>>
        self._list_cli = self._node.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        while not self._list_cli.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info(
                "Waiting for /controller_manager/list_controllers service..."
            )

        self._node.get_logger().info(
            "Connected to /controller_manager/list_controllers service."
        )

        # <<< Initialize /controller_manager/list_controllers <<<

        self._status = {
            "scaled_joint_trajectory_controller": False,
            "forward_velocity_controller": False,
        }

        # Get the status of the controllers

    def get_controller_status(self):
        req = ListControllers.Request()

        response: ListControllers.Response = self._list_cli.call(req)

        if response is not None:
            for controller in response.controller:
                controller: ControllerState

                for name in self._status.keys():
                    if controller.name == name:
                        self._status[name] = controller.state == "active"

        else:
            self._node.get_logger().error("Failed to get controller list.")

        return self._status

    def change_controller_state(self, data: dict):
        request = SwitchController.Request()

        request.activate_controllers = data.get("activate_controllers", [])
        request.deactivate_controllers = data.get("deactivate_controllers", [])

        self._node.get_logger().info(
            f"Activating controllers: {request.activate_controllers}, Deactivating controllers: {request.deactivate_controllers}"
        )

        request.strictness = SwitchController.Request.STRICT

        response: SwitchController.Response = self._switch_cli.call(request)

        if response is not None:
            if response.ok:
                return True

        return False


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

    @property
    def J(self) -> np.ndarray:
        jacobian = self._compute_jacobian()

        if jacobian is None:
            return None

        if np.isclose(np.linalg.norm(jacobian), 0.0):
            self._node.get_logger().warn(
                "Jacobian is zero, cannot compute joint commands!"
            )
            return None

        return jacobian

    @staticmethod
    def dh_transform(a, d, alpha, theta):
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

    def forward_kinematics(self) -> PoseStamped:
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

        # 변환된 자코비안
        T_eef_corrected = T_correction @ T_eef

        # ----------------- Pose 메시지로 변환 -----------------
        rotation_matrix = T_eef_corrected[:3, :3]
        qx, qy, qz, qw = rotutils.quaternion_from_rotation_matrix(
            rotation_matrix=rotation_matrix
        )

        result = PoseStamped(
            header=Header(
                frame_id="world", stamp=self._node.get_clock().now().to_msg()
            ),
            pose=Pose(
                position=Point(
                    x=T_eef_corrected[0, 3],
                    y=T_eef_corrected[1, 3],
                    z=T_eef_corrected[2, 3],
                ),
                orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
            ),
        )

        return result

    def _compute_jacobian(self) -> np.ndarray:
        joint_states: JointState = self._joint_states_manager.data

        if joint_states is None:
            self._node.get_logger().warn("JointState data not available yet.")
            return None

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

        T = np.eye(4)
        origins = [T[:3, 3]]
        z_axes = [T[:3, 2]]

        # 누적 T, z축, origin
        for i in range(6):
            a, d, alpha = self._dh_params[i]
            Ti = EEFManager.dh_transform(a, d, alpha, q[i])
            T = T @ Ti
            origins.append(T[:3, 3])
            z_axes.append(T[:3, 2])

        # gripper_link offset
        T_grip = np.eye(4)
        T_grip[:3, 3] = self._gripper_offset  # Adjusted for gripper offset
        T = T @ T_grip

        o_e = T[:3, 3]

        J = np.zeros((6, 6))
        for i in range(6):
            zi = z_axes[i]
            oi = origins[i]
            J[:3, i] = np.cross(zi, o_e - oi)
            J[3:, i] = zi

        T_correction = np.eye(6)
        T_correction[0, 0] = -1  # x 반전
        T_correction[1, 1] = -1  # y 반전

        # 변환된 자코비안
        J_corrected = T_correction @ J

        return J_corrected


class PI_Controller(object):
    def __init__(self, kp: float, ki: float, dt: float, max_error: float = 1.0):
        self._kp = kp
        self._ki = ki
        self._dt = dt

        self._max_error = max_error
        self._last_error = 0.0  # Last error value

        self._integral = 0.0

    def compute(self, error: float) -> float:
        """
        Compute the control output using PI control.
        """

        integral = 0.0

        # If the error is same as the last error, do not update the integral
        if error != self._last_error:
            integral = error * self._dt

        self._integral = np.clip(
            self._integral + integral, -self._max_error, self._max_error
        )

        self._last_error = error

        return np.clip(
            self._kp * error + self._ki * self._integral,
            -self._max_error,
            self._max_error,
        )


class PathTrackingManager(object):
    def __init__(self, node: Node):
        self._node = node

        self._pi_controller = PI_Controller(
            kp=1.0,  # Proportional gain
            ki=0.1,  # Integral gain
            dt=0.1,  # Time step
            max_error=np.pi / 4.0,  # Maximum error (45 degrees/sec)
        )

        self._path: np.ndarray = None
        self._directions: np.ndarray = None

    @staticmethod
    def parse_np_path_to_ros_path(path: np.ndarray, z: float, header: Header) -> Path:
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

    @staticmethod
    def generate_parabolic_path(
        x_start, y_start, x_offset=0.05, y_offset=-0.2, num_points=100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        x, y: 시작점 좌표
        num_points: 보간 점 개수
        x_offset: x의 최대 변위 (포물선 폭)
        """

        # y 값 보간 (선형)
        y_end = y_start + y_offset
        y_values = np.linspace(y_start, y_end, num_points)

        # 포물선 중심 y (x 최대값을 가지는 지점)
        y_mid = (y_start + y_end) / 2.0

        # 포물선 계수 a 계산: x = -a * (y - y_mid)^2 + x + x_offset
        a = x_offset / ((y_start - y_mid) ** 2)

        # x 값 생성 (포물선)
        x_values = -a * (y_values - y_mid) ** 2 + x_start + x_offset

        # path 생성
        path = np.vstack((x_values, y_values)).T

        # 방향 벡터 계산
        directions = np.zeros_like(path)
        directions[1:-1] = path[2:] - path[:-2]
        directions[0] = path[1] - path[0]
        directions[-1] = path[-1] - path[-2]

        # 단위 벡터화
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms

        return path, directions

    @staticmethod
    def linear_path(
        x_start: float,
        y_start: float,
        x_offset: float,
        y_offset: float,
        num_points: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a linearly interpolated path from (x_start, y_start)
        to (x_start + x_offset, y_start + y_offset) in a vectorized manner.

        Args:
            x_start (float): Starting x-coordinate.
            y_start (float): Starting y-coordinate.
            x_offset (float): Displacement in x from start to end.
            y_offset (float): Displacement in y from start to end.
            num_points (int): Number of points to generate along the path.

        Returns:
            np.ndarray: Array of shape (num_points, 2), each row [x_i, y_i].
        """
        if num_points < 1:
            return np.empty((0, 2))

        # Parameter t goes from 0 to 1 in num_points steps
        t = np.linspace(0.0, 1.0, num_points)

        # Vectorized interpolation
        x_coords = x_start + t * x_offset
        y_coords = y_start + t * y_offset

        # Stack into (num_points, 2)
        path = np.stack((x_coords, y_coords), axis=1)

        # 방향 벡터 계산
        directions = np.zeros_like(path)
        directions[1:-1] = path[2:] - path[:-2]
        directions[0] = path[1] - path[0]
        directions[-1] = path[-1] - path[-2]

        # 단위 벡터화
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms

        return path, directions

    @staticmethod
    def get_closest_index(
        last_index: int, window_size: int, path: np.ndarray, position: np.ndarray
    ) -> int:
        low = max(0, last_index - window_size)
        high = min(len(path), last_index + window_size)

        dists = np.linalg.norm(path[low:high] - position, axis=1)
        idx_rel = np.argmin(dists)
        idx = low + idx_rel

        # distances = np.linalg.norm(path - position, axis=1)
        # idx = np.argmin(distances)

        return idx

    @staticmethod
    def get_target_point_and_direction(
        path: np.ndarray, direction: np.ndarray, index: int, lookup_index: int = 1
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Returns the target point and direction at the given index.
        """
        if path is None or index < 0 or index >= len(path):
            raise IndexError("Path not created or index out of range.")

        if index + lookup_index > len(path) - 1:
            return path[index], direction[index]
        else:
            return (path[index + lookup_index], direction[index + lookup_index])

    def tracking(self, current_position: np.ndarray, current_direction: np.ndarray):
        # 1. Calculate the closest point on the path. and get the index
        closest_index = np.argmin(np.linalg.norm(self._path - current_position, axis=1))

        # 2. Get the target point and direction
        _, target_direction = self.get_target_point_and_direction(
            closest_index, lookup_index=1
        )

        # 3. Calculate the angle error
        current_angle = np.arctan2(
            current_direction[1], current_direction[0]
        )  # Current direction angle

        target_angle = np.arctan2(
            target_direction[1], target_direction[0]
        )  # Target direction angle

        angle_error = target_angle - current_angle

        # 4. Calculate the control output using the PI controller
        control_output = self._pi_controller.compute(angle_error)

        # 5. Return the control output
        return control_output


class Status(Enum):
    WAITING = 0
    PLANNING = 1
    EXECUTING = 2
    SWEEPING = 3
    HOMING = 4
    END = 5
    TEST = 999
    TESTEND = 1000


class MainNode(Node):
    def __init__(self):
        super().__init__("robot_control_node")

        # >>> State Machine Initialization >>>
        self._status = Status.WAITING
        self._methods = {
            Status.WAITING: self._waiting,
            Status.PLANNING: self._planning,
            Status.EXECUTING: self._executing,
            Status.SWEEPING: self._sweeping_velocity,
            Status.HOMING: self._homing,
            Status.END: self._end,
            Status.TEST: self._test,  # Test state for debugging
            Status.TESTEND: self._end,  # Test end state for debugging
        }
        # <<< State Machine Initialization <<<

        # >>> Potential Field Path Planner >>>
        self._pf_planner = PotentialFieldPlanner(
            rr=0.08,  # robot radius [m]
            resolution=0.02,  # grid resolution [m]
            kp=1.0,  # attractive potential gain
            eta=100.0,  # repulsive potential gain
            area_offset=1.0,  # potential area width [m]
            oscillations=3,  # number of previous positions to check for oscillations
        )
        # <<< Potential Field Path Planner <<<

        # >>> MoveIt2 Service Managers >>>
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
        # <<< MoveIt2 Service Managers <<<

        # >>> Manager >>>
        self._object_transform_manager = SimObjectTransformManager(self)  # Sim Object
        self._controller_switcher = ControllerSwitcher(node=self)
        self._path_tracking_manager = PathTrackingManager(node=self)
        self._eef_manager = EEFManager(node=self)
        self._joint_state_manager = SimpleSubscriberManager(
            node=self, topic_name="/joint_states", msg_type=JointState
        )
        # <<< Manager <<<

        # >>> Parameters >>>
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
        self._home_pose: PoseStamped = None
        self._pfp_path: np.ndarray = None
        self._z0 = 0.2549  # Initial Z position of the end effector
        self._z = self._z0 + 0.1
        # <<< Parameters <<<

        # >>> ROS Publishers >>>
        self._marker_publisher = self.create_publisher(
            MarkerArray,
            self.get_name() + "/obstacle_markers",
            qos_profile=qos_profile_system_default,
        )
        self._path_publisher = self.create_publisher(
            Path,
            self.get_name() + "/planned_path",
            qos_profile=qos_profile_system_default,
        )
        self._cmd_publication = self.create_publisher(
            Float64MultiArray,
            "/forward_velocity_controller/commands",
            qos_profile=qos_profile_system_default,
        )

    def run(self):
        # self.get_logger().info(f"Running state machine in status: {self._status.name}")
        try:
            self._methods[self._status]()
        except ValueError as ve:
            self.get_logger().warn(f"Error in state machine: {ve}")
            return None
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")
            return None

    def _update_status(self):
        # 1. Update the status based on the current mode
        self._status = Status(self._status.value + 1)

        if self._status == Status.END:
            self.get_logger().info("Ending the state machine.")
            return None

        if self._status == Status.TESTEND:
            self.get_logger().info("Test end state reached.")
            return None

        # 2. Check current mode
        if self._status == Status.EXECUTING or self._status == Status.HOMING:
            self._controller_switcher.change_controller_state(
                {
                    "activate_controllers": ["scaled_joint_trajectory_controller"],
                    "deactivate_controllers": ["forward_velocity_controller"],
                }
            )

        elif self._status == Status.SWEEPING:
            self._controller_switcher.change_controller_state(
                {
                    "activate_controllers": ["forward_velocity_controller"],
                    "deactivate_controllers": ["scaled_joint_trajectory_controller"],
                }
            )

        return self._status.value

    # >>> State Machine Methods
    def _waiting(self):
        """
        1. Get FK Position
        """
        if self._joint_state_manager.data is None:
            self.get_logger().warn("Joint states not available yet.")
            return None

        self.get_logger().info("Calculating FK Pose...")

        self._home_pose: PoseStamped = self._fk_manager.run(
            # joint_states=self._joint_state_manager.joint_states,
            joint_states=self._home_joints,  # Use home joints for FK
            end_effector="gripper_link",
        )

        self.get_logger().info(
            f"FK Pose: {self._home_pose.pose.position.x:.3f}, {self._home_pose.pose.position.y:.3f}, {self._home_pose.pose.position.z:.3f}"
        )

        if self._home_pose is None:
            raise ValueError("FK Pose could not be calculated.")

        self.get_logger().info("Calculating Path to Home Position...")

        goal_constraints = self._kinematic_path_manager.get_goal_constraint(
            goal_joint_states=self._home_joints, tolerance=0.05
        )

        traj: RobotTrajectory = self._kinematic_path_manager.run(
            goal_constraints=[goal_constraints],
            path_constraints=None,
            joint_states=self._joint_state_manager.data,
            num_planning_attempts=50,
        )

        # traj: RobotTrajectory = self._cartesian_path_manager.run(
        #     header=Header(frame_id="world", stamp=self.get_clock().now().to_msg()),
        #     waypoints=[self._home_pose.pose],
        #     joint_states=self._joint_state_manager.joint_states,
        #     end_effector="gripper_link",
        # )

        self.get_logger().info("Path to Home Position calculated.")

        if traj is None:
            raise ValueError("Trajectory could not be calculated.")

        scaled_traj = self._execute_trajectory_manager.scale_trajectory(
            trajectory=traj,
            scale_factor=0.7,  # Scale down the trajectory for FK pose
        )

        self.get_logger().info("Running Trajectory to Home Position...")

        self._execute_trajectory_manager.run(
            trajectory=scaled_traj,
        )

        self._update_status()

        return True

    def _planning(self):
        """
        2. Plan Path
        - Start: fk_pose
        - Goal: user-defined goal pose
        - Obstacles: user-defined obstacles
        """
        if self._object_transform_manager.data is None:
            # ONLY FOR ISSAC SIM
            raise ValueError("Object transform data not available.")

        # Start and Goal Points for Potential Field Planner
        start_point = PotentialPoint(
            x=self._home_pose.pose.position.x,
            y=self._home_pose.pose.position.y,
        )
        goal_point = PotentialPoint(
            x=self._object_transform_manager.data.pose.position.x,
            y=self._object_transform_manager.data.pose.position.y + 0.1,
        )
        obstacles = [
            PotentialObstacle(
                x=self._object_transform_manager.data.pose.position.x,
                y=self._object_transform_manager.data.pose.position.y,
                r=0.03,
            ),
        ]

        self.get_logger().info("Planning Path with Potential Field Planner...")
        self.get_logger().info(f"Start Point: {start_point.x:.3f}, {start_point.y:.3f}")
        self.get_logger().info(f"Goal Point: {goal_point.x:.3f}, {goal_point.y:.3f}")
        for obs_id in range(len(obstacles)):
            self.get_logger().info(
                f"Obstacle Point: {obstacles[obs_id].x:.3f}, {obstacles[obs_id].y:.3f}, Radius: {obstacles[obs_id].r:.3f}"
            )

        self._pfp_path: np.ndarray = self._pf_planner.planning(
            start=start_point,
            goal=goal_point,
            obstacles=obstacles,
        )

        self.get_logger().info(
            f"Planning Finished! - Path: {len(self._pfp_path)} points."
        )

        # Publish the path and obstacles (Visualization)
        obstacle_msg: MarkerArray = self._pf_planner.parse_obstacles_to_marker_array(
            obstacles,
            self._home_pose.pose.position.z,
            Header(frame_id="world", stamp=self.get_clock().now().to_msg()),
        )
        path_msg: Path = self._pf_planner.parse_np_path_to_path(
            self._pfp_path,
            self._z,
            Header(frame_id="world", stamp=self.get_clock().now().to_msg()),
        )

        for _ in range(10):
            self._marker_publisher.publish(obstacle_msg)
            self._path_publisher.publish(path_msg)

        self._update_status()

    def _executing(self):
        """
        3. Execute Path (Cartesian Path)
        """
        sliced_path = np.vstack(
            [self._pfp_path[::50], self._pfp_path[-1]]
        )  # Slicing the path for execution

        poses: List[Pose] = self._pf_planner.parse_np_path_to_pose_array_with_z_curve(
            sliced_path,
            start_z=self._home_pose.pose.position.z,
            end_z=self._z,  # Assuming the end Z is the same as the initial Z
            orientation=self._home_pose.pose.orientation,
        )

        self.get_logger().info(
            f"Calculating Cartesian Path with {len(poses)} waypoints..."
        )

        traj: RobotTrajectory = self._cartesian_path_manager.run(
            header=Header(frame_id="world", stamp=self.get_clock().now().to_msg()),
            waypoints=poses,
            end_effector="gripper_link",
            joint_states=self._joint_state_manager.data,
        )

        last_traj: JointTrajectoryPoint = traj.joint_trajectory.points[-1]

        execution_time = last_traj.time_from_start
        execution_time_float = execution_time.sec + execution_time.nanosec * 1e-9

        target_execution_time = 5.0  # User-defined target execution time in seconds
        executution_time_ratio = execution_time_float / target_execution_time
        # executution_time_ratio = 0.2  # User-defined scale factor for execution time

        if traj is None:
            raise ValueError("Trajectory could not be calculated.")

        self.get_logger().info(
            f"Trajectory calculated with {len(traj.joint_trajectory.points)} points."
        )
        self.get_logger().info(
            f"Scaling the Trajectory - Execution Time: {execution_time_float:.2f} seconds, Scale Factor: {executution_time_ratio:.2f}"
        )

        scaled_traj = self._execute_trajectory_manager.scale_trajectory(
            trajectory=traj,
            scale_factor=executution_time_ratio,
        )

        self.get_logger().info("Running Scaled Trajectory...")

        self._execute_trajectory_manager.run(
            trajectory=scaled_traj,
        )

        self._update_status()

    def _sweeping(self):
        # Current EEF pose
        start_pose: PoseStamped = self._fk_manager.run(
            joint_states=self._joint_state_manager.data,
            end_effector="gripper_link",
        )

        # END pose for sweeping : 0.3m right from the start pose
        end_pose = PoseStamped(
            header=start_pose.header,
            pose=Pose(
                position=Point(
                    x=start_pose.pose.position.x,
                    y=start_pose.pose.position.y - 0.3,
                    z=start_pose.pose.position.z,
                ),
                orientation=start_pose.pose.orientation,
            ),
        )

        self.get_logger().info(
            f"Start Pose: {start_pose.pose.position.x:.3f}, {start_pose.pose.position.y:.3f}, {start_pose.pose.position.z:.3f}"
        )
        self.get_logger().info(
            f"End Pose: {end_pose.pose.position.x:.3f}, {end_pose.pose.position.y:.3f}, {end_pose.pose.position.z:.3f}"
        )
        self.get_logger().info("Calculating Cartesian Path for Sweeping...")

        traj: RobotTrajectory = self._cartesian_path_manager.run(
            header=Header(frame_id="world", stamp=self.get_clock().now().to_msg()),
            waypoints=[start_pose.pose, end_pose.pose],
            joint_states=self._joint_state_manager.data,
            end_effector="gripper_link",
        )

        if traj is None:
            raise ValueError("Trajectory could not be calculated for sweeping.")

        scaled_traj = self._execute_trajectory_manager.scale_trajectory(
            trajectory=traj,
            scale_factor=0.5,  # Scale down the trajectory for sweeping
        )

        self.get_logger().info(
            f"Trajectory calculated with {len(traj.joint_trajectory.points)} points."
        )
        self.get_logger().info(f"Executing the scaled trajectory for sweeping")

        self._execute_trajectory_manager.run(
            trajectory=scaled_traj,
        )

        self.get_logger().info("Sweeping completed successfully.")

        self._update_status()

    def _homing(self):
        self.get_logger().info("Calculating Homing Trajectory...")

        traj: RobotTrajectory = self._kinematic_path_manager.run(
            goal_constraints=[
                self._kinematic_path_manager.get_goal_constraint(
                    goal_joint_states=self._home_joints, tolerance=0.05
                )
            ],
            path_constraints=None,
            joint_states=self._joint_state_manager.data,
            num_planning_attempts=100,
            allowed_planning_time=3.0,
            max_velocity_scaling_factor=1.0,
            max_acceleration_scaling_factor=1.0,
        )

        # traj: RobotTrajectory = self._cartesian_path_manager.run(
        #     header=Header(frame_id="world", stamp=self.get_clock().now().to_msg()),
        #     waypoints=[self._home_pose.pose],
        #     joint_states=self._joint_state_manager.data,
        #     end_effector="gripper_link",
        # )

        if traj is None:
            raise ValueError("Trajectory could not be calculated for homing.")

        self.get_logger().info(
            f"Trajectory calculated with {len(traj.joint_trajectory.points)} points."
        )

        self.get_logger().info("Calculating scaled trajectory for homing...")

        scaled_traj = self._execute_trajectory_manager.scale_trajectory(
            trajectory=traj,
            scale_factor=0.7,  # Scale down the trajectory for homing
        )

        self.get_logger().info("Running scaled trajectory for homing...")

        self._execute_trajectory_manager.run(
            trajectory=scaled_traj,
        )

        self.get_logger().info("Homing completed successfully.")

        self._update_status()

    def _end(self):
        pass

    def _test(self):
        pass

    def _sweeping_velocity(self):
        """
        4. Sweeping with velocity control
        """
        object_pose: PoseStamped = self._object_transform_manager.data

        if object_pose is None:
            raise ValueError("Object pose data not available.")

        # path, direction = self._path_tracking_manager.generate_parabolic_path(
        #     x_start=object_pose.pose.position.x,
        #     y_start=object_pose.pose.position.y,
        #     x_offset=0.03,  # Offset for the parabolic path
        #     y_offset=-0.25,  # Offset for the parabolic path
        #     num_points=100,  # Number of points in the path
        # )

        path, direction = self._path_tracking_manager.linear_path(
            x_start=object_pose.pose.position.x,
            y_start=object_pose.pose.position.y,
            x_offset=-0.05,  # Offset for the linear path
            y_offset=-0.25,  # Offset for the linear path
            num_points=100,  # Number of points in the path
        )

        path_msg = self._path_tracking_manager.parse_np_path_to_ros_path(
            path=path,
            z=self._z,  # Assuming a flat plane at z = 0.2549 + 0.1
            header=Header(frame_id="world", stamp=self.get_clock().now().to_msg()),
        )
        for _ in range(10):
            self._path_publisher.publish(path_msg)

        hz = 30.0
        dt = 1.0 / hz  # Time step for the controller

        target_idx = 0
        last_idx = len(path) - 1
        object_poses = deque(
            maxlen=5
        )  # Store last 5 object poses for direction calculation
        object_pose: PoseStamped = self._object_transform_manager.data
        object_poses.append(object_pose)

        rate = self.create_rate(hz)

        while rclpy.ok():
            eef_pose: PoseStamped = self._eef_manager.forward_kinematics()
            object_pose: PoseStamped = self._object_transform_manager.data
            object_np_pose = np.array(
                [
                    object_pose.pose.position.x,
                    object_pose.pose.position.y,
                ]
            )

            target_idx = PathTrackingManager.get_closest_index(
                last_index=target_idx,
                window_size=10,  # Search window size for closest point
                path=path,
                position=np.array(
                    [object_pose.pose.position.x, object_pose.pose.position.y]
                ),
            )
            target_np_pose, target_np_direction = (
                PathTrackingManager.get_target_point_and_direction(
                    path=path, direction=direction, index=target_idx, lookup_index=1
                )
            )

            target_pose = PoseStamped(
                pose=Pose(
                    position=Point(x=target_np_pose[0], y=target_np_pose[1], z=self._z),
                    orientation=Quaternion(
                        x=0.0,
                        y=0.0,
                        z=0.0,
                        w=1.0,  # Assuming no rotation for simplicity
                    ),
                )
            )
            past_object_pose: PoseStamped = object_poses[0]

            object_direction = np.array(
                [
                    object_pose.pose.position.x - past_object_pose.pose.position.x,
                    object_pose.pose.position.y - past_object_pose.pose.position.y,
                ]
            )

            object_angle = np.arctan2(object_direction[1], object_direction[0])
            target_angle = np.arctan2(target_np_direction[1], target_np_direction[0])
            eef_angle, _, _ = rotutils.euler_from_quaternion(
                x=eef_pose.pose.orientation.x,
                y=eef_pose.pose.orientation.y,
                z=eef_pose.pose.orientation.z,
                w=eef_pose.pose.orientation.w,
            )

            is_contact = np.linalg.norm(object_direction) > 0.0003

            angle_diff = (target_angle - object_angle) if is_contact else 0.0

            object_poses.append(object_pose)

            # print(target_idx, target_pose.pose.position.x, eef_pose.pose.position.x)
            x_gain = 1.5

            x_diff = target_pose.pose.position.x - eef_pose.pose.position.x
            x_control_msg = np.array(
                [x_diff * x_gain, 0.0, 0.0, 0.0, 0.0, 0.0]
            )  # Move towards the object in x direction

            y_gain = 1.5

            y_diff = target_pose.pose.position.y - eef_pose.pose.position.y
            y_control_msg = np.array([0.0, y_diff * y_gain, 0.0, 0.0, 0.0, 0.0])

            cte_vector = (
                object_np_pose - target_np_pose
            )  # currnet position - target position
            cte = np.dot(cte_vector, target_np_direction)  # Cross-track error

            cte_gain = 2.0 if is_contact else 0.0
            cte_control_msg = np.clip(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, cte * cte_gain]), -0.2, 0.2
            )

            angle_gain = (
                1.0 if np.abs(eef_angle) < np.deg2rad(60) and is_contact else 0.0
            )
            angle_control_msg = np.clip(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, angle_diff * angle_gain]),
                -0.2,
                0.2,
            )  # TODO: REMOVE 0.0

            control_msg = y_control_msg + x_control_msg
            normalized_control_msg = (
                ((control_msg / np.linalg.norm(control_msg)) * 0.03)
                + angle_control_msg
                + cte_control_msg
            )

            # >>> Make Joint Control Message >>>
            J_inv = np.linalg.pinv(self._eef_manager.J)
            joint_control_msg = (J_inv @ normalized_control_msg).tolist()

            self._cmd_publication.publish(Float64MultiArray(data=joint_control_msg))

            self.get_logger().info(f"CMD: {control_msg}")

            if target_idx == last_idx:
                self.get_logger().info(
                    f"Reached the end of the path at index {target_idx}."
                )

                for _ in range(30):
                    self._cmd_publication.publish(
                        Float64MultiArray(data=[0.0] * 6)
                    )  # Stop the robot

                self._update_status()
                break

            rate.sleep()

    # <<< State Machine Methods


def main():
    rclpy.init(args=None)

    node = MainNode()

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 10.0  # Frequency in Hz
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
