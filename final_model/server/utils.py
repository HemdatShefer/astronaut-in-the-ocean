import pickle
import struct
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from temp.temp import temp


@dataclass
class HandleImageData:
    temp: str
    # image_bytes: bytes

def dummy():
    pass

def handle_client(client):
    while True:
        img = receive_img(client)
        print(temp(img))

#todo check if it works
def receive_img(conn):
    # Receive image size first (4 bytes, network byte order)
    size_data = conn.recv(4)
    if len(size_data) < 4:
        raise ValueError("Did not receive image size properly")
    img_size = struct.unpack('!I', size_data)[0]
    print(f"Expecting {img_size} bytes")

    # Receive the actual image bytes
    img_bytes = b''
    while len(img_bytes) < img_size:
        packet = conn.recv(4096)
        if not packet:
            break
        img_bytes += packet

    print(f"Received {len(img_bytes)} bytes")

    # Convert bytes to image
    img = bytes_to_image(img_bytes)
    return img

def handle_img(data):
    print(data)


def bytes_to_image(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img