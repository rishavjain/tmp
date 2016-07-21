from server.start_server import startserver

from multiprocessing import Process
import sys
import socket
import time

def client(HOST, PORT, data):

    # Create a socket (SOCK_STREAM means a TCP socket)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((HOST, PORT))
        sock.sendall(bytes(data + "\n", "utf-8"))

        # Receive data from the server and shut down
        received = str(sock.recv(1024), "utf-8")

    print("Sent:     {}".format(data))
    print("Received: {}".format(received))


if __name__ == '__main__':
    p = Process(target=startserver, args=())

    p.start()

    # time.sleep(2)

    f = open('tmp/server-address')
    HOST, PORT = f.readline(), int(f.readline())
    f.close()

    client(HOST, PORT, 'a')
    client(HOST, PORT, 'b')
    client(HOST, PORT, 'c')

    p.terminate()
