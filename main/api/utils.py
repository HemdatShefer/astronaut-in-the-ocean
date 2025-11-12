import struct

import cv2
import numpy as np


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


def bytes_to_image(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img