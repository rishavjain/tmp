from server.start_server import startserver

from multiprocessing import Process


if __name__ == '__main__':
    p = Process(target=startserver, args=())

    p.start()



    p.join()
