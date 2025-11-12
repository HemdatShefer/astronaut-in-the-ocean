from utils import receive_img
from utils import handle_result
from temp.temp import temp


def handle_client(client):
    while True:
        img = receive_img(client)
        result = temp(img)
        handle_result(client, result)