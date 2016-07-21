import socketserver
import socket
import time
import random

PORT = 9999


class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        # just send back the same data, but upper-cased
        self.request.sendall(self.data.upper())


def startserver():
    HOST = socket.gethostbyname(socket.gethostname())

    # Create the server, binding to localhost on port 9999
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)

    print('server listening at {}:{}'.format(HOST, PORT))

    f = open('tmp/server-address', 'w')
    f.write(HOST)
    f.write('\n')
    f.write(str(PORT))
    f.close()

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
