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
from tf2_msgs.msg import TFMessage as TFM

# TF
from tf2_ros import *

# Python
import numpy as np


class ObjectTransformManager:
    def __init__(self, node: Node):
        self._node = node

        self._sub = self._node.create_subscription(
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

            self._data = PoseStamped(
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

            print(self._data)

            self._publish_data(self._data)


def main():
    rclpy.init(args=None)

    node = Node("object_transform_manager")
    om = ObjectTransformManager(node=node)

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 1.0  # Frequency in Hz
    rate = node.create_rate(hz)

    try:
        while rclpy.ok():
            # node.run()
            rate.sleep()

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
