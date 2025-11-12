import json
import struct
from datetime import datetime

import cv2
import numpy as np

def handle_result(client, result):
    data = build_response_json(client, result)
    client.sendall(data.encode())

def receive_img(client):
    size_data = client.recv(4)
    if len(size_data) < 4:
        raise ValueError("Did not receive image size properly")

    img_size = struct.unpack('!I', size_data)[0]

    img_bytes = b''
    while len(img_bytes) < img_size:
        packet = client.recv(4096)
        if not packet:
            break
        img_bytes += packet

    img = bytes_to_image(img_bytes)
    return img

#private functions
def bytes_to_image(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img

def build_response_json(client, result):
    detections = []
    for box in result[0].boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        detections.append({
            "class": result[0].names[cls_id],
            "confidence": float(box.conf[0]),
            "bbox": [x1, y1, x2, y2]
        })

    return json.dumps({
        "client": getattr(client, "name", "Unknown"),
        "timestamp": datetime.now().isoformat(),
        "detections_count": len(detections),
        "detections": detections
    })