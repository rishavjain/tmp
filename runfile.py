from server.start_server import run_server

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
        print("Sent:     {}".format(data))

    #     # Receive data from the server and shut down
    #     received = str(sock.recv(1024), "utf-8")
    #
    # print("Received: {}".format(received))


if __name__ == '__main__':

    serverProcess = Process(target=run_server, args=())

    # Exit the server thread when the main thread terminates
    serverProcess.daemon = True
    serverProcess.start()

    print("Server running in thread:", serverProcess.name)

    time.sleep(2)

    f = open('tmp/server-address')
    HOST, PORT = f.readline().strip(), int(f.readline().strip())
    f.close()

    client(HOST, PORT, 'a')
    client(HOST, PORT, 'b')
    client(HOST, PORT, 'c')

    serverProcess.join()