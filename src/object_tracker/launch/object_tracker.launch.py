from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 런치 인자 선언
    model_file_arg = DeclareLaunchArgument(
        "model_file",
        description="Path to the YOLOv8 model file",
        default_value="best_model.pt",
    )

    grid_data_file_arg = DeclareLaunchArgument(
        "grid_data_file",
        description="Path to the grid data file",
        default_value="grid_data.json",
    )

    obj_bounds_file_arg = DeclareLaunchArgument(
        "obj_bounds_file",
        description="Path to the object bounds file",
        default_value="obj_bounds.json",
    )

    conf_threshold_arg = DeclareLaunchArgument(
        "conf_threshold",
        description="Confidence threshold for object detection",
        default_value="0.7",
    )

    is_test_arg = DeclareLaunchArgument(
        "debug", default_value="false", description="Test Bench Mode"
    )

    segmentation_node = Node(
        package="object_tracker",
        executable="real_time_segmentation_node",
        name="real_time_segmentation_node",
        output="screen",
        arguments=[
            "--model_file",
            LaunchConfiguration("model_file"),
            "--obj_bounds_file",
            LaunchConfiguration("obj_bounds_file"),
            "--conf_threshold",
            LaunchConfiguration("conf_threshold"),
        ],
    )

    pose_estimation_node = Node(
        package="object_tracker",
        executable="pointcloud_pose_estimation_server",
        name="pointcloud_pose_estimation_server",
        output="screen",
        arguments=[
            "--grid_data_file",
            LaunchConfiguration("grid_data_file"),
            "--obj_bounds_file",
            LaunchConfiguration("obj_bounds_file"),
            "--debug",
            LaunchConfiguration("debug"),
        ],
    )

    return LaunchDescription(
        [
            model_file_arg,
            grid_data_file_arg,
            obj_bounds_file_arg,
            conf_threshold_arg,
            is_test_arg,
            pose_estimation_node,
            segmentation_node,
        ]
    )
