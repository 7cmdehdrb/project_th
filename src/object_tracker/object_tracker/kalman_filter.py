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
from builtin_interfaces.msg import Time as BuiltinTime

# TF
from tf2_ros import *

# Python
import numpy as np
from typing import Tuple, Optional, List, Union, Any
from base_package.manager import TransformManager


class PoseManager:
    def __init__(self, node: Node):
        self._node = node

        # >>> Initialize state variables >>>
        self._z = np.array([0.0, 0.0, 0.0, 0.0])
        self._dt = 0.1

        # >>> Initialize Pose Subscriber >>>
        self._pose_subscriber = self._node.create_subscription(
            PoseStamped,
            "/megapose_client_node/pose",  # TODO: Set the topic name
            self._pose_callback,
            qos_profile=qos_profile_system_default,
        )
        self._pose = PoseStamped()

        self._frame_id = ""
        self._last_time: Time = None

    @property
    def frame_id(self) -> str:
        """
        Returns the frame ID of the last received pose message.
        """
        return self._frame_id

    @property
    def dt(self) -> float:
        """
        Returns the time step in seconds.
        """
        return self._dt

    @property
    def z(self) -> np.ndarray:
        """
        Returns the measurement vector [x, y, vx, vy].
        """
        return self._z

    @property
    def last_time(self) -> Time | None:
        """
        Returns the last time a pose message was received.
        If no pose message has been received, returns None.
        """
        return self._last_time

    def _pose_callback(self, msg: PoseStamped):
        """
        Callback for receiving pose messages.
        Updates the measurement vector and time step.
        """
        current_time = self._node.get_clock().now()

        # 1. Initialize the time step
        if self._last_time is None:
            self._frame_id = msg.header.frame_id
            self._last_time = current_time

            self._node.get_logger().info(
                "First pose message received, initializing time."
            )

            return None

        self._dt = (
            current_time - self._last_time
        ).nanoseconds / 1e9  # Time step in seconds

        vx = (msg.pose.position.x - self._pose.pose.position.x) / self._dt
        vy = (msg.pose.position.y - self._pose.pose.position.y) / self._dt

        self._z = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                vx,
                vy,
            ]
        )  # Measurement vector [x, y, vx, vy]

        self._pose = msg  # Update the last pose message
        self._last_time = current_time  # Update the last time to the current time


class KalmanFilterNode(Node):
    class KalmanState:
        def __init__(self):
            self._x = np.array([0.0, 0.0, 0.0, 0.0])  # State vector [x, y, vx, vy]
            self._P = np.eye(4)  # State

            self._dt = 0.1

            self._A = np.array(
                [
                    [1.0, 0, self._dt, 0.0],
                    [0.0, 1.0, 0.0, self._dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            self._Q = np.eye(4) * 0.01  # Process noise covariance
            self._R = (
                np.eye(4) * 0.01
            )  # Measurement noise covariance TODO: Set appropriate values
            self._H = np.eye(4)  # Measurement matrix

        @property
        def x(self) -> np.ndarray:
            """
            Returns the current state vector [x, y, vx, vy].
            """
            return self._x

        def update_A(self, dt: float) -> np.ndarray:
            """
            Update the state transition matrix A based on the time step dt.

            Args:
                dt: Time step in seconds.
            """
            self._dt = dt
            self._A = np.array(
                [
                    [1.0, 0, self._dt, 0.0],
                    [0.0, 1.0, 0.0, self._dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            return self._A

        def filter(self, z: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray]:
            """
            Perform a Kalman filter step
            with a new measurement z.
            Args:
                z: Measurement vector.
            Returns:
                updated_x: Updated state vector.
                updated_P: Updated covariance matrix.
            """
            if z is None:
                self._x, self._P = self._predict()  # Predict without update
                return self._x, self._P

            predicted_x, predicted_P = self._predict()
            self._x, self._P = self._update(
                z, predicted_x, predicted_P
            )  # Auto-update self._x and self._P

            return self._x, self._P

        def _predict(self) -> Tuple[np.ndarray, np.ndarray]:
            """
            Predict the next state and covariance.

            Returns:
                predicted_x: Predicted state vector.
                predicted_P: Predicted covariance matrix.
            """
            predicted_x = self._A @ self._x
            predicted_P = self._A @ self._P @ self._A.T + self._Q

            return predicted_x, predicted_P

        def _update(self, z, predicted_x, predicted_P) -> Tuple[np.ndarray, np.ndarray]:
            """
            Update the state with a new measurement.

            Args:
                z: Measurement vector.
                predicted_x: Predicted state vector.
                predicted_P: Predicted covariance matrix.
            Returns:
                updated_x: Updated state vector.
                updated_P: Updated covariance matrix.
            """
            S = self._H @ self._P @ self._H.T + self._R
            K = self._P @ self._H.T @ np.linalg.pinv(S)

            x = predicted_x + K @ (z - self._H @ predicted_x)
            P = predicted_P - K @ self._H @ predicted_P

            return x, P

    def __init__(self):
        super().__init__("kalman_filter_node")

        self._state = KalmanFilterNode.KalmanState()
        self._pose_manager = PoseManager(node=self)

        self._tf_manager = TransformManager(node=self)

        self._pose_publisher = self.create_publisher(
            PoseStamped,
            self.get_name() + "/pose",
            qos_profile=qos_profile_system_default,
        )
        self._observation_publisher = self.create_publisher(
            Float32MultiArray,
            self.get_name() + "/observation",
            qos_profile=qos_profile_system_default,
        )
        self._status_subscriber = self.create_subscription(
            UInt8,
            "/megapose_client_node/status",
            self._status_callback,
            qos_profile=qos_profile_system_default,
        )
        self._megapose_status = 0

    def _status_callback(self, msg: UInt8):
        self._megapose_status = int(msg.data)

    def _publish_pose(self):
        msg = PoseStamped(
            header=Header(
                frame_id=self._pose_manager.frame_id,
                stamp=self.get_clock().now().to_msg(),
            ),
            pose=Pose(
                position=Point(
                    x=self._state.x[0],
                    y=self._state.x[1],
                    z=0.0,  # TODO: Set the z-coordinate if needed
                ),
                orientation=Quaternion(
                    x=0.0,  # TODO: Set the orientation if needed
                    y=0.0,
                    z=0.0,
                    w=1.0,  # Default orientation (no rotation)
                ),
            ),
        )

        self._pose_publisher.publish(msg)

    def _publish_observation(self):
        transformed_pose: PoseStamped = self._tf_manager.transform_pose(
            PoseStamped(
                header=Header(
                    frame_id=self._pose_manager.frame_id,
                    stamp=self.get_clock().now().to_msg(),
                ),
                pose=Pose(
                    position=Point(
                        x=self._state.x[0],
                        y=self._state.x[1],
                        z=0.0,  # TODO: Set the z-coordinate if needed
                    ),
                    orientation=Quaternion(
                        x=0.0,  # TODO: Set the orientation if needed
                        y=0.0,
                        z=0.0,
                        w=1.0,  # Default orientation (no rotation)
                    ),
                ),
            ),
            target_frame="world",
            source_frame=self._pose_manager.frame_id,
        )

        if transformed_pose is not None:
            observation_msg = Float32MultiArray(
                data=[
                    transformed_pose.pose.position.x,
                    transformed_pose.pose.position.y,
                    0.2549,
                ]
            )
            self._observation_publisher.publish(observation_msg)

    def run(self):
        if self._pose_manager.last_time is None:
            self.get_logger().warn("No pose message received yet, skipping run.")
            return

        # 1. Update A
        self._state.update_A(dt=self._pose_manager._dt)

        # 2. Perform the Kalman filter step
        z = (
            self._pose_manager.z if self._megapose_status == 2 else None
        )  # Use the measurement if the time step is reasonable
        x, _ = self._state.filter(z)

        self.get_logger().info(
            f"KF: x={x[0]:.2f}, y={x[1]:.2f}, vx={x[2]:.2f}, vy={x[3]:.2f}"
        )

        # 3. Publish the pose
        self._publish_pose()
        self._publish_observation()


def main():
    rclpy.init(args=None)

    node = KalmanFilterNode()

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 15.0  # Frequency in Hz
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
