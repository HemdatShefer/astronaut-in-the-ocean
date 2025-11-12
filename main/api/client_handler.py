from main.api.utils import receive_img


def handle_client(client):
    while True:
        img = receive_img(client)
        # result = model(img)
        # handle_result(result)