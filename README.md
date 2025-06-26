# Switch Controllers

## Turn on forward_position_controller & Turn off scaled_joint_trajectory_controller

```bash
ros2 control switch_controllers --deactivate scaled_joint_trajectory_controller --activate forward_position_controller
```
```bash
ros2 control switch_controllers --activate scaled_joint_trajectory_controller --deactivate forward_velocity_controller
```

## Activate Trigger

```bash
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger
```

## Publish Servo Message

```bash
ros2 topic pub /servo_node/delta_twist_cmds geometry_msgs/msg/TwistStamped "{ header: { stamp: 'now', frame_id: 'base_link' },  twist: {linear: {x: -0.1}, angular: {  }}}" -r 10
```

<hr>

# HOW TO LAUNCH?

## 0. Launch RVIZ2

```bash
rviz2 -d rviz.rviz
```

## 1. Launch Camera

```bash
ros2 launch realsense2_camera rs_launch.py camera_name:="camera1" pointcloud.enable:=true rgb_camera.color_profile:="640,480,30" depth_module.depth_profile:="640,480,30" rgb_camera.enable_auto_exposure:=false rgb_camera.exposure:="100" usb_port_id:="6-1.1"
```

## 1-1. Launch Side Camera

```bash
ros2 launch realsense2_camera rs_launch.py camera_name:="camera2" pointcloud.enable:=true rgb_camera.color_profile:="640,480,30" depth_module.depth_profile:="640,480,30" rgb_camera.enable_auto_exposure:=false rgb_camera.exposure:="100" usb_port_id:="6-1.3"
```

## 2. Launch UR Bringup

```bash
ros2 launch ur_bringup ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.2.2 launch_rviz:=false
```
```bash
ros2 launch ur_bringup ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.56.101 launch_rviz:=false
```

## 3. Launch UR Moveit

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e
```

```bash
ros2 launch ur_moveit_with_gripper move_group.launch.py
ros2 launch ur_moveit_with_gripper  rsp.launch.py
```

## 4. Launch YOLO Segmentation

```bash
python3 /home/irol/workspace/project_th/src/object_tracker/object_tracker/real_time_segmentation_node.py --model_file /home/irol/workspace/project_th/src/object_tracker/resource/best_hg_0428.pt --conf_threshold 0.7
```

## 5. Launch Megapose Client Node

```bash
python3 /home/irol/workspace/project_th/src/object_tracker/object_tracker/megapose_client.py --refiner_iterations 1 --score_threshold 0.2 --target_object alive
```

# TF

```bash
ros2 run tf2_ros static_transform_publisher -0.2575 -0.2725 0.46 0.0 0.0 0.0 0.1 world camera1_joint1
```

```bash
ros2 run tf2_ros static_transform_publisher 0.05 0.0 0.0 0.0 0.0 0.258819 0.9659258  camera1_joint1 camera1_link
```
