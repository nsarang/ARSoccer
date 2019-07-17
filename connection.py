import socket
import threading
import time


class DataReciever:
    def __init__(self, HOST, PORT):
        self.HOST = HOST
        self.PORT = PORT
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((HOST, PORT))
        self.socket.listen(10)
        self.lock = threading.Lock()
        self.x = None
        self.y = None
        threading.Thread(target=self._recieve, args=()).start()

    def _recieve(self):
        while True:
            (conn, addr) = self.socket.accept()
            with conn:
                print ('Connected by', addr)
                while True:
                    msg = conn.recv(7)
                    if not msg:
                        break
                    msg = msg.decode('ascii')
                    with self.lock:
                    	self.x, self.y = (int(msg[:3]), int(msg[5:]))
                    # print(self.x, self.y)

    def get_cords(self):
        with self.lock:
            return self.x, self.y


class DataSender:
    def __init__(self, HOST, PORT):
        self.HOST = HOST
        self.PORT = PORT
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.HOST, self.PORT))

    def send(self, velocity, angle):
    	msg = '{0:03d}${1:03d}'.format(int(velocity), int(angle))
    	print(msg)
    	self.socket.send(bytes(msg, 'utf-8'))
	
    def close(self):
    	self.socket.close()