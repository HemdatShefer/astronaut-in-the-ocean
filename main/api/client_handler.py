from utils import receive_img
from temp.temp import temp


def handle_client(client):
    while True:
        img = receive_img(client)
        result = temp(img)
        print(result)
        # handle_result(result)