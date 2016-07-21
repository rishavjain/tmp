import socket
import threading
import socketserver
import time
import random


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        print("connected to {}:{}".format(self.client_address[0], self.client_address[1]))
        time.sleep(5*random.random())
        data = str(self.request.recv(1024), 'ascii')
        print("{}: recieved: {}".format(threading.current_thread().name, data))

        # cur_thread = threading.current_thread()
        # response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')
        # self.request.sendall(response)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


def startserver():
    # Port 0 means to select an arbitrary unused port
    HOST = socket.gethostbyname(socket.gethostname())

    server = ThreadedTCPServer((HOST, 0), ThreadedTCPRequestHandler)
    ip, port = server.server_address

    f = open('tmp/server-address', 'w')
    f.write(ip)
    f.write('\n')
    f.write(str(port))
    f.close()

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    print("Server running in thread:", server_thread.name)

    server_thread.join()
