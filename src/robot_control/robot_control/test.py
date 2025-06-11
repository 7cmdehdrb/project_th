# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from builtin_interfaces.msg import Duration as BuiltinDuration
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from control_msgs.msg import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# TF
from tf2_ros import *

# Python
import numpy as np


class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")

        # Example of creating a publisher
        self._pub1 = self.create_publisher(
            JointJog,
            "/servo_node/delta_joint_cmds",
            qos_profile=qos_profile_system_default,
        )
        self._pub2 = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            qos_profile=qos_profile_system_default,
        )
        self._pub3 = self.create_publisher(
            Float64MultiArray,
            "/forward_velocity_controller/commands",
            qos_profile=qos_profile_system_default,
        )

        self._subscriber = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            qos_profile=qos_profile_system_default,
        )
        self._joint_states: JointState = None

    def _joint_state_callback(self, msg: JointState):
        self._joint_states = msg

    def _generate_velocity_profile(self, p0, p1, v0, v1, T, num_points=10):
        """
        보간된 속도 함수 v(t)를 생성.
        - p0, p1: 시작/종료 위치
        - v0, v1: 시작/종료 속도
        - T: 총 시간
        - 반환: 시간 배열 t, 속도 배열 v, 위치 배열 p (적분값)
        """
        dp = p1 - p0

        d = v0
        A = np.array([[T**3, T**2, T], [T**4 / 4, T**3 / 3, T**2 / 2]])
        b_vec = np.array([v1 - v0, dp - v0 * T])

        coeffs = np.linalg.lstsq(A, b_vec, rcond=None)[0]  # a, b, c
        a, b_, c = coeffs

        # t grid
        t = np.linspace(0, T, num_points)
        v = a * t**3 + b_ * t**2 + c * t + d

        return t, v

    def run2(self):
        if self._joint_states is None:
            return

        current_joint = self._joint_states.position
        target_joint = [
            current_joint[0] + np.deg2rad(10.0) / 10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        current_velocity = self._joint_states.velocity
        target_velocity = [
            np.deg2rad(10.0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        ts = []
        vs = []

        # Generate velocity profile for each joint
        ts, vs = self._generate_velocity_profile(
            p0=cj, p1=tj, v0=cv, v1=tv, T=0.1, num_points=10
        )

        print("Time steps:", ts[0], ts[-1])
        print("Velocities:", vs[0], vs[-1])

        print(target_velocity)

        data = Float64MultiArray()

        data.data = [
            np.random.uniform(-0.2, 0.2) for _ in range(len(self._joint_states.name))
        ]

        # print(data)

        # self._pub3.publish(data)


def main():
    rclpy.init(args=None)

    node = TestNode()

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 10.0  # Frequency in Hz
    rate = node.create_rate(hz)

    try:
        while rclpy.ok():
            node.run2()
            rate.sleep()

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
