import sys
import os
import socket
import struct
from omni.isaac.core.prims import XFormPrim
from pxr import UsdPhysics, Usd, UsdGeom


class SocketClient:
    def __init__(self, host="127.0.0.1", port=9000):
        self.host = host
        self.port = port
        self.header = b"HEAD"
        self.footer = b"TAIL"
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.client_socket.connect((self.host, self.port))
        print(f"[Client] Connected to {self.host}:{self.port}")

    def send(self, float_array):
        if len(float_array) != 12:
            raise ValueError("Array must be of length 12.")
        payload = struct.pack("!12f", *float_array)
        packet = self.header + payload + self.footer
        self.client_socket.sendall(packet)
        print(f"[Client] Sent: {float_array}")

    def receive(self):
        expected_size = 56  # 4 bytes header + 48 bytes payload + 4 bytes footer
        packet = self.client_socket.recv(expected_size)
        if packet.startswith(self.header) and packet.endswith(self.footer):
            payload = packet[4:-4]
            data = struct.unpack("!12f", payload)
            # print(f"[Client] Received: {data}")
            return data
        else:
            print("[Client] Invalid packet received.")
            return None


prim_paths = [
    "/World/ur5e/base_link_inertia/shoulder_pan_joint",
    "/World/ur5e/shoulder_link/shoulder_lift_joint",
    "/World/ur5e/upper_arm_link/elbow_joint",
    "/World/ur5e/forearm_link/wrist_1_joint",
    "/World/ur5e/wrist_1_link/wrist_2_joint",
    "/World/ur5e/wrist_2_link/wrist_3_joint",
]

prims = [XFormPrim(path) for path in prim_paths]
target_positions_attrs = [
    prim.prim.GetAttribute("drive:angular:physics:targetPosition") for prim in prims
]
currnet_positions_attrs = [
    prim.prim.GetAttribute("state:angular:physics:position") for prim in prims
]
current_velocities_attrs = [
    prim.prim.GetAttribute("state:angular:physics:velocity") for prim in prims
]

client = SocketClient(host="220.149.84.111", port=9000)
client.connect()

# Example usage
try:
    for _ in range(10000):  # Simulate sending and receiving data multiple times
        # Simulate sending data

        current_positions = [attr.Get() for attr in currnet_positions_attrs]
        current_velocities = [attr.Get() for attr in current_velocities_attrs]
        current_state = current_positions + current_velocities

        client.send(current_state)

        # Simulate receiving data
        received_data = client.receive()
        if received_data:
            print(f"[Client] Received data: {received_data}")
            for i in range(6):
                target_positions_attrs[i].Set(received_data[i])
                current_velocities_attrs[i].Set(received_data[i + 6])

except KeyboardInterrupt:
    print("[Client] Shutting down.")
finally:
    client.client_socket.close()
