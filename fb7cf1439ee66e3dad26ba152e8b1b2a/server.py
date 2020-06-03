from socket import *
from time import ctime

HOST = ''
PORT = 21567
BUFSIZ = 1024
ADDR = (HOST, PORT)

tcpServerSock = socket(AF_INET, SOCK_STREAM)
tcpServerSock.bind(ADDR)
tcpServerSock.listen(5)

while True:
    print('waiting for connection...')
    tcpClientSock, addr = tcpServerSock.accept()
    print('...connected from:', addr)

    while True:
        data = tcpClientSock.recv(BUFSIZ)
        if not data:
            break
        tcpClientSock.send(bytes('[%s] %s' % (ctime(), data.decode('utf-8')), 'utf-8'))

    tcpClientSock.close()
tcpServerSock.close()