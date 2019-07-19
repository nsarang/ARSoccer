import socket
import threading
import time
from collections import namedtuple


Stat = namedtuple("Stat", 'x y v', defaults=(None,) * 3)



class DataReciever:
    def __init__(self, HOST, PORT):
        self.HOST = HOST
        self.PORT = PORT
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((HOST, PORT))
        self.socket.listen(10)
        self.lock = threading.Lock()
        self.cur_stat = Stat()
        self.prev_stat = Stat()
        threading.Thread(target=self._recieve, args=()).start()

    def _recieve(self):
        while True:
            (conn, addr) = self.socket.accept()
            with conn:
                print('Connected by', addr)
                while True:
                    msg = conn.recv(9)
                    if not msg:
                        break
                    msg = msg.decode('ascii')
                    self.prev_stat = self.cur_stat
                    with self.lock:
                    	self.cur_stat = Stat(x=int(msg[:3]),
                                             y=int(msg[3:6]),
                                             v=int(msg[6:]))

    def get_stats(self):
        dx = self.cur_stat.x - self.prev_stat.x
        dy = self.cur_stat.y - self.prev_stat.y
        with self.lock:
            return (*self.cur_stat, dx, dy)


class DataSender:
    def __init__(self, HOST, PORT):
        self.HOST = HOST
        self.PORT = PORT
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        

    def send(self, velocity, angle):
    	if not self.connected:
    		try:
    			self.socket.connect((self.HOST, self.PORT))
    			self.connected = True
    		except:
    			return
    	msg = '{0:03d}${1:03d}'.format(int(velocity), int(angle))
    	self.socket.send(bytes(msg, 'utf-8'))
	
    def close(self):
    	self.socket.close()