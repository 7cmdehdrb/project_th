# Python
import os
import sys
import numpy as np
import socket
import struct


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
        if len(float_array) != 6:
            raise ValueError("Array must be of length 6.")
        payload = struct.pack("!6f", *float_array)
        packet = self.header + payload + self.footer
        self.client_socket.sendall(packet)
        # print(f"[Client] Sent: {float_array}")

    def receive(self):
        expected_size = 32
        packet = self.client_socket.recv(expected_size)
        if packet.startswith(self.header) and packet.endswith(self.footer):
            payload = packet[4:-4]
            data = struct.unpack("!6f", payload)
            # print(f"[Client] Received: {data}")
            return data
        else:
            # print("[Client] Invalid packet received.")
            return None


client = None


def setup(db: og.Database):
    global client
    client = None
    client = SocketClient(host="220.149.84.111", port=9000)
    client.connect()


def cleanup(db: og.Database):
    global client
    print("Cleaning up socket client...")
    client.client_socket.close()
    pass


def compute(db: og.Database):
    global client
    if client is None:
        print("Socket client not initialized.")
        return False

    current_joint: np.ndarray = db.inputs.input_var

    # Simulate sending data
    client.send(np.deg2rad(current_joint).tolist())

    # Simulate receiving data
    received_data = client.receive()

    if received_data:
        db.outputs.output_var = np.array(received_data)
    return True
