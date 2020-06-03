from socket import *
import base64

HOST = '127.0.0.1'
PORT = 21567
BUFSIZ = 1024
ADDR = (HOST, PORT)

tcpCliSock = socket(AF_INET, SOCK_STREAM)
tcpCliSock.connect(ADDR)

while True:
    data = input('Client Text: \n')
    if not data:
        break
    tcpCliSock.send(bytes(data, 'utf-8'))
    data = tcpCliSock.recv(BUFSIZ)
    data = base64.encodestring(data)

    print('\nEncrypted: ', data.decode('utf-8'))
    if not data:
        break
    op = input('Need Decrypted(Yes/No)\n? ')
    if op == "Yes":
    	data = base64.decodestring(data)
    	print('\nText: ', data.decode('utf-8'))
    #decode
