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


def main():
    client = SocketClient(host="220.149.84.101", port=9000)
    client.connect()

    # Example usage
    try:
        for _ in range(10000):  # Simulate sending and receiving data multiple times
            # Simulate sending data
            client.send([0.0] * 12)  # Sending 12 floats
            # Simulate receiving data
            received_data = client.receive()
            if received_data:
                print(f"Processed data: {received_data}")
    except KeyboardInterrupt:
        print("[Client] Shutting down.")
    finally:
        client.client_socket.close()


if __name__ == "__main__":
    main()
