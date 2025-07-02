# ============================================================================= #type: ignore  # noqa E501
# Copyright © 2025 NaturalPoint, Inc. All Rights Reserved.
#
# THIS SOFTWARE IS GOVERNED BY THE OPTITRACK PLUGINS EULA AVAILABLE AT https://www.optitrack.com/about/legal/eula.html #type: ignore  # noqa E501
# AND/OR FOR DOWNLOAD WITH THE APPLICABLE SOFTWARE FILE(S) (“PLUGINS EULA”). BY DOWNLOADING, INSTALLING, ACTIVATING #type: ignore  # noqa E501
# AND/OR OTHERWISE USING THE SOFTWARE, YOU ARE AGREEING THAT YOU HAVE READ, AND THAT YOU AGREE TO COMPLY WITH AND ARE #type: ignore  # noqa E501
# BOUND BY, THE PLUGINS EULA AND ALL APPLICABLE LAWS AND REGULATIONS. IF YOU DO NOT AGREE TO BE BOUND BY THE PLUGINS #type: ignore  # noqa E501
# EULA, THEN YOU MAY NOT DOWNLOAD, INSTALL, ACTIVATE OR OTHERWISE USE THE SOFTWARE AND YOU MUST PROMPTLY DELETE OR #type: ignore  # noqa E501
# RETURN IT. IF YOU ARE DOWNLOADING, INSTALLING, ACTIVATING AND/OR OTHERWISE USING THE SOFTWARE ON BEHALF OF AN ENTITY, #type: ignore  # noqa E501
# THEN BY DOING SO YOU REPRESENT AND WARRANT THAT YOU HAVE THE APPROPRIATE AUTHORITY TO ACCEPT THE PLUGINS EULA ON #type: ignore  # noqa E501
# BEHALF OF SUCH ENTITY. See license file in root directory for additional governing terms and information. #type: ignore  # noqa E501
# ============================================================================= #type: ignore  # noqa E501


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish
# a connection and receive data via that NatNet connection
# to decode it using the NatNetClientLibrary.

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

# TF
from tf2_ros import *

# Python
import numpy as np
import threading
from typing import Dict, Any, Optional, Callable

# NatNet
import sys
import time
from NatNetClient import NatNetClient
import DataDescriptions
import MoCapData


# This is a callback function that gets connected to the NatNet client
# and called once per mocap frame.


def receive_new_frame(data_dict):
    order_list = [
        "frameNumber",
        "markerSetCount",
        "unlabeledMarkersCount",  # type: ignore  # noqa F841
        "rigidBodyCount",
        "skeletonCount",
        "labeledMarkerCount",
        "timecode",
        "timecodeSub",
        "timestamp",
        "isRecording",
        "trackedModelsChanged",
    ]
    dump_args = False
    if dump_args is True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "= "
            if key in data_dict:
                out_string += data_dict[key] + " "
            out_string += "/"
        print(out_string)


def receive_new_frame_with_data(data_dict):
    order_list = [
        "frameNumber",
        "markerSetCount",
        "unlabeledMarkersCount",  # type: ignore  # noqa F841
        "rigidBodyCount",
        "skeletonCount",
        "labeledMarkerCount",
        "timecode",
        "timecodeSub",
        "timestamp",
        "isRecording",
        "trackedModelsChanged",
        "offset",
        "mocap_data",
    ]
    dump_args = True
    if dump_args is True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "= "
            if key in data_dict:
                out_string += str(data_dict[key]) + " "
            out_string += "/"
        print(out_string)


def request_data_descriptions(s_client: NatNetClient):
    # Request the model definitions
    s_client.send_request(s_client.command_socket, s_client.NAT_REQUEST_MODELDEF, "", (s_client.server_ip_address, s_client.command_port))  # type: ignore  # noqa F501


class NatNetClientNode(Node):
    def __init__(self, target_id: int):
        super().__init__("natnet_client_node")

        self._target_id = target_id

        self._pose_publisher = self.create_publisher(
            PoseStamped,
            self.get_name() + "/pose",
            qos_profile=qos_profile_system_default,
        )

    def _publish_pose(self, pose_msg: PoseStamped):
        self._pose_publisher.publish(pose_msg)

    def receive_rigid_body_frame(
        self,
        new_id: str,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float, float],
    ):
        if int(new_id) != self._target_id:
            return None

        pose_msg = PoseStamped(
            header=Header(
                frame_id="optitrack_world", stamp=self.get_clock().now().to_msg()
            ),
            pose=Pose(
                position=Point(**dict(zip(["x", "y", "z"], position))),
                orientation=Quaternion(**dict(zip(["x", "y", "z", "w"], rotation))),
            ),
        )
        self._publish_pose(pose_msg)


def main(arg=None):
    rclpy.init(args=arg)

    node = NatNetClientNode(target_id=1)

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    optionsDict = {
        "clientAddress": "192.168.50.251",
        "serverAddress": "192.168.50.7",
        "use_multicast": False,
        "stream_type": "d",
    }

    streaming_client = NatNetClient()

    # streaming_client.new_frame_with_data_listener = receive_new_frame_with_data  # type ignore # noqa E501
    streaming_client.new_frame_listener = receive_new_frame
    streaming_client.rigid_body_listener = node.receive_rigid_body_frame

    streaming_client.set_use_multicast(optionsDict["use_multicast"])
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.run(optionsDict["stream_type"])

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
