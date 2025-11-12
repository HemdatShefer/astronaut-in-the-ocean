from .model.detect_ships import detect_ships
from .utils import *


def handle_client(client):
    while True:
        img = receive_img(client)
        result = detect_ships(img)
        handle_result(client, result["detections"])