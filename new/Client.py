import zmq
import json

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://10.2.125.73:12345')
socket.setsockopt(zmq.SUBSCRIBE,''.encode('utf-8'))

if __name__ == '__main__':
    print('客户端开启，等待服务端结果...')
    while 1:
        msg = socket.recv_json()
        data = json.loads(msg)
        print(data['score'])
        print(data['result'])
