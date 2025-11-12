import socket
import cv2
import time

UDP_IP = "127.0.0.1"   # Change to server IP if remote
UDP_PORT = 5005
CHUNK_SIZE = 60000
CHUNK_END = b"END"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(10)

# Load the image
img = cv2.imread("ocean-with-boat.jpg")
if img is None:
    raise FileNotFoundError("Image file not found!")

# Encode as JPEG
_, encoded = cv2.imencode(".jpg", img)
data = encoded.tobytes()
print(f"[CLIENT] Image size: {len(data)} bytes")

# Send image in chunks
for i in range(0, len(data), CHUNK_SIZE):
    chunk = data[i:i+CHUNK_SIZE]
    sock.sendto(chunk, (UDP_IP, UDP_PORT))
    time.sleep(0.001)  # small delay helps prevent packet loss

# Send end marker
sock.sendto(CHUNK_END, (UDP_IP, UDP_PORT))
print("[CLIENT] Image sent, waiting for reply...")

# Wait for reply
try:
    reply, _ = sock.recvfrom(1024)
    print("[CLIENT] Server replied:", reply.decode())
except socket.timeout:
    print("[CLIENT] No reply received (timeout).")
