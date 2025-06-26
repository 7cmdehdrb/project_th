from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="moveit_servo",
                executable="servo_node_main",
                name="servo_server",
                output="screen",
                parameters=[
                    "/home/min/7cmdehdrb/project_th/src/ur_moveit/ur_moveit_with_gripper/config/ur_servo.yaml"
                ],
            )
        ]
    )
