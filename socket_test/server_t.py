import socket
import struct
import threading


class SocketServer:
    def __init__(self, host="127.0.0.1", port=9000):
        self.host = host
        self.port = port
        self.header = b"HEAD"
        self.footer = b"TAIL"
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_conn = None

    def start(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"[Server] Listening on {self.host}:{self.port} ...")
        self.client_conn, addr = self.server_socket.accept()
        print(f"[Server] Connection established with {addr}")

    def send(self, float_array):
        if len(float_array) != 12:
            raise ValueError("Array must be of length 12.")
        payload = struct.pack("!12f", *float_array)
        packet = self.header + payload + self.footer
        self.client_conn.sendall(packet)
        # print(f"[Server] Sent: {float_array}")

    def receive(self):
        expected_size = 56  # 4 + 48 + 4
        packet = self.client_conn.recv(expected_size)
        if packet.startswith(self.header) and packet.endswith(self.footer):
            payload = packet[4:-4]
            data = struct.unpack("!12f", payload)
            # print(f"[Server] Received: {data}")
            return data
        else:
            print("[Server] Invalid packet received.")
            return None


def main():
    server = SocketServer(host="0.0.0.0", port=9000)
    server.start()

    import time

    last_time = time.time()

    # Example usage
    try:
        while True:
            # Simulate sending data
            server.send([0.0] * 12)
            # Simulate receiving data
            received_data = server.receive()
            if received_data:
                current_time = time.time()
                elapsed_time = current_time - last_time
                last_time = current_time
                print(f"[Server] Elapsed time: {elapsed_time:.5f} seconds")
                print(f"[Server] Processed data: {received_data}")
    except KeyboardInterrupt:
        print("[Server] Shutting down.")
    finally:
        server.client_conn.close()
        server.server_socket.close()


if __name__ == "__main__":
    main()
