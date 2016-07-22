import socket
import threading
import socketserver
import time
import random
from multiprocessing import Process

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        tmp = 5*random.random();
        print("connected to {}:{}, {}".format(self.client_address[0], self.client_address[1], tmp))
        time.sleep(tmp)

        data = str(self.request.recv(1024), 'ascii')
        print("{}: recieved: {}".format(threading.current_thread().name, data))

        # cur_thread = threading.current_thread()
        # response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')
        # self.request.sendall(response)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


def run_server():
    # Port 0 means to select an arbitrary unused port
    HOST = socket.gethostbyname(socket.gethostname())

    server = ThreadedTCPServer((HOST, 0), ThreadedTCPRequestHandler)
    ip, port = server.server_address

    f = open('tmp/server-address', 'w')
    f.write(ip)
    f.write('\n')
    f.write(str(port))
    f.close()

    server.serve_forever()
