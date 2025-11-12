import socket
import threading

from final_model.server.utils import handle_client

HOST = 'localhost'
PORT = 9000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

clients = set()

print(f"Server listening on {HOST}:{PORT}")

try:
    while True:
        client, addr = server_socket.accept()
        print(f"Connected by {addr}")
        clients.add(client)
        clientThread = threading.Thread(target=handle_client, args=(client,))
        clientThread.start()
finally:
    print("Closing all client sockets...")
    for c in clients:
        c.close()
    server_socket.close()
    print("Server socket closed.")
