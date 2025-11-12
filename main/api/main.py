import socket

from main.api.client_handler import handle_client

HOST = 'localhost'
PORT = 9000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Server listening on {HOST}:{PORT}")

clients = []
try:
    client, addr = server_socket.accept()
    clients.append(client)
    handle_client(client)
finally:
    print("Closing server...")
    for c in clients:
        c.close()
    server_socket.close()