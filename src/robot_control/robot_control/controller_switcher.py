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
from controller_manager_msgs.srv import SwitchController, ListControllers
from controller_manager_msgs.msg import ControllerState

# TF
from tf2_ros import *

# Python
import numpy as np


class ControllerSwitcher(Node):
    def __init__(self):
        super().__init__("controller_switcher")

        # >>> Initialize /controller_manager/switch_controller >>>
        self._switch_cli = self.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        while not self._switch_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Waiting for /controller_manager/switch_controller service..."
            )

        self.get_logger().info(
            "Connected to /controller_manager/switch_controller service."
        )

        # <<< Initialize /controller_manager/switch_controller <<<

        # >>> Initialize /controller_manager/list_controllers >>>
        self._list_cli = self.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        while not self._list_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Waiting for /controller_manager/list_controllers service..."
            )

        self.get_logger().info(
            "Connected to /controller_manager/list_controllers service."
        )

        # <<< Initialize /controller_manager/list_controllers <<<

        self._status = {
            "scaled_joint_trajectory_controller": False,
            "forward_position_controller": False,
        }

        # Get the status of the controllers
        self.get_controller_status()

    def get_controller_status(self):
        req = ListControllers.Request()

        future = self._list_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        response: ListControllers.Response = future.result()

        if response is not None:
            for controller in response.controller:
                controller: ControllerState

                for name in self._status.keys():
                    if controller.name == name:
                        self._status[name] = controller.state == "active"

        else:
            self.get_logger().error("Failed to get controller list.")

        return self._status

    def change_controller_state(self):
        request = SwitchController.Request()

        scaled_joint_trajectory_controller = self._status[
            "scaled_joint_trajectory_controller"
        ]

        request.activate_controllers = (
            ["scaled_joint_trajectory_controller"]
            if not scaled_joint_trajectory_controller
            else ["forward_position_controller"]
        )
        request.deactivate_controllers = (
            ["forward_position_controller"]
            if not scaled_joint_trajectory_controller
            else ["scaled_joint_trajectory_controller"]
        )

        request.strictness = SwitchController.Request.STRICT

        future = self._switch_cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response: SwitchController.Response = future.result()

        if response is not None:
            if response.ok:
                self.get_logger().info("Successfully switched controllers.")

                self._status = {
                    "scaled_joint_trajectory_controller": not scaled_joint_trajectory_controller,
                    "forward_position_controller": scaled_joint_trajectory_controller,
                }

                return True
            else:
                self.get_logger().error("Failed to switch controllers.")

        else:
            self.get_logger().error("Service call failed.")

        return False


def main():
    rclpy.init()

    node = ControllerSwitcher()

    node.get_controller_status()
    node.change_controller_state()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
