```bash
ros2 control switch_controllers --deactivate scaled_joint_trajectory_controller --activate forward_position_controller
```

RESTART
```bash
ros2 control switch_controllers --activate scaled_joint_trajectory_controller --deactivate forward_position_controller
```

```bash
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger
```

```bash
ros2 topic pub /servo_node/delta_twist_cmds geometry_msgs/msg/TwistStamped "{ header: { stamp: 'now', frame_id: 'world' },  twist: {linear: {x: -0.1}, angular: {  }}}" -r 10
```

```bash
ros2 launch ur_movei
t_config ur_moveit.launch.py ur_type:=ur5e use_fake_hardware:=true
```