import socket
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------
# UDP server configuration
# -----------------------
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
CHUNK_END = b"END"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"[SERVER] Listening on {UDP_IP}:{UDP_PORT}")

buffer = b""
client_addr = None

# -----------------------
# Receive one image
# -----------------------
while True:
    data, addr = sock.recvfrom(65535)
    client_addr = addr

    if data == CHUNK_END:
        print(f"[SERVER] Received full image ({len(buffer)} bytes) from {addr}")
        # Reply to client
        sock.sendto(b"hi", client_addr)
        print("[SERVER] Replied 'hi' to client. Closing server socket...")
        break
    else:
        buffer += data

sock.close()
print("[SERVER] Socket closed.")

# -----------------------
# Decode the image
# -----------------------
np_data = np.frombuffer(buffer, np.uint8)
img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
if img is None:
    print("[SERVER] Failed to decode image.")
    exit()

# -----------------------
# Run YOLO detection
# -----------------------
model = YOLO("yolov8n.pt")  # You can use yolov8n.pt, yolov8s.pt, etc.

results = model(img)

# Show annotated image and print coordinates
for r in results:
    r.show()  # displays the image with bounding boxes
    for box in r.boxes:
        cls = model.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"{cls}: {conf:.2f}  [{x1}, {y1}, {x2}, {y2}]")
