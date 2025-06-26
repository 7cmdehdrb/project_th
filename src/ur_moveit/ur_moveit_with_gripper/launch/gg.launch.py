import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node


def load_file_from_src(absolute_file_path):
    with open(absolute_file_path, "r") as f:
        return f.read()


def load_yaml_from_src(absolute_file_path):
    with open(absolute_file_path, "r") as f:
        return yaml.safe_load(f)


def generate_launch_description():
    # src 절대 경로 base
    base_path = "/home/min/7cmdehdrb/project_th/src/ur_moveit/ur_moveit_with_gripper"

    # 올바른 URDF / SRDF / kinematics.yaml / servo.yaml 절대 경로
    urdf_path = os.path.join(base_path, "config/ur5e.urdf.xacro")
    srdf_path = os.path.join(base_path, "config/ur5e.srdf")
    kinematics_path = os.path.join(base_path, "config/kinematics.yaml")
    servo_yaml_path = os.path.join(base_path, "config/ur_servo.yaml")
    ros2_controllers_yaml = os.path.join(base_path, "config/ros2_controllers.yaml")
    rviz_config_path = os.path.join(base_path, "config/ur_servo_demo.rviz")

    # 파일 로드
    urdf_content = load_file_from_src(urdf_path)
    srdf_content = load_file_from_src(srdf_path)
    kinematics_yaml = load_yaml_from_src(kinematics_path)
    servo_yaml = load_yaml_from_src(servo_yaml_path)

    # 파라미터 dict
    robot_description = {"robot_description": urdf_content}
    robot_description_semantic = {"robot_description_semantic": srdf_content}
    robot_description_kinematics = {"robot_description_kinematics": kinematics_yaml}
    servo_params = {"moveit_servo": servo_yaml}

    # ros2_control node
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, ros2_controllers_yaml],
        output="screen",
    )

    # joint_state_broadcaster
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    # scaled_joint_trajectory_controller
    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "scaled_joint_trajectory_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    # Servo node
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        parameters=[
            servo_params,
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
        ],
        output="screen",
    )

    # RViz node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config_path],
        parameters=[robot_description, robot_description_semantic],
        output="screen",
    )

    return LaunchDescription(
        [
            ros2_control_node,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            servo_node,
            rviz_node,
        ]
    )
