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
from builtin_interfaces.msg import Duration as DurationMsg

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

        self._markers = MarkerArray()

        self._marker_array_publisher = self.create_publisher(
            MarkerArray,
            self.get_name() + "/marker_array",
            qos_profile=qos_profile_system_default,
        )

        self._hz = 30.0  # Frequency of publishing markers
        self._timer = self.create_timer(self._hz, self._publish_marker_array)

    def _get_pose_msg(
        self,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float, float],
    ) -> PoseStamped:
        pose_msg = PoseStamped(
            header=Header(
                frame_id="optitrack_world", stamp=self.get_clock().now().to_msg()
            ),
            pose=Pose(
                position=Point(**dict(zip(["x", "y", "z"], position))),
                orientation=Quaternion(**dict(zip(["x", "y", "z", "w"], rotation))),
            ),
        )

        return pose_msg

    def _get_marker_msg(
        self,
        new_id: str,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float, float],
    ) -> Marker:
        marker_msg = Marker(
            header=Header(
                frame_id="optitrack_world", stamp=self.get_clock().now().to_msg()
            ),
            ns=new_id,
            id=int(new_id),
            type=Marker.CUBE,
            action=Marker.ADD,
            pose=Pose(
                position=Point(**dict(zip(["x", "y", "z"], position))),
                orientation=Quaternion(**dict(zip(["x", "y", "z", "w"], rotation))),
            ),
            scale=Vector3(x=0.1, y=0.1, z=0.1),
            color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            lifetime=DurationMsg(sec=0, nanosec=100000000),  # 0.1 seconds
        )
        return marker_msg

    def _update_marker_array(self, maker_msg: Marker):
        # Check if the marker already exists in the array
        for i, existing_marker in enumerate(self._markers.markers):
            existing_marker: Marker

            if existing_marker.id == maker_msg.id:
                # Update the existing marker
                self._markers.markers[i] = maker_msg
                return True  # Marker updated

        # If not found, add the new marker to the array
        self._markers.markers.append(maker_msg)

    def _publish_marker_array(self):
        if self._marker_array_publisher.get_subscription_count() > 0:
            self._marker_array_publisher.publish(self._markers)

    def receive_rigid_body_frame(
        self,
        new_id: str,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float, float],
    ):
        # Transform the position and rotation into a PoseStamped message
        marker_msg = self._get_marker_msg(new_id, position, rotation)

        # Update the marker array with the new marker
        self._update_marker_array(marker_msg)

        # Publish the marker array
        self._publish_marker_array()


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
