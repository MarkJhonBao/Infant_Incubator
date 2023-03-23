# Run with two terminals
import selectors            #实现socket高并发的selectors库
import socket
import time

mysel = selectors.DefaultSelector()
keep_running = True

def read(connection, mask):
    "读取事件的回调"
    global keep_running

    client_address = connection.getpeername()       #连接到当前套接字的远端的地址
    print('read({})'.format(client_address))
    data = connection.recv(1024)    #接收TCP数据
    if data:
        # 可读的客户端 socket 有数据
        print('  received {!r}'.format(data))

        if data.decode() != "exit":
            connection.sendall("i am server".encode())      ##发送完整的TCP数据(本质就是循环调用send,sendall在待发送数据量大于己端缓存区剩余空间时,数据不丢失,循环调用send直到发完)
            print("send")
        else:
            # 将空结果解释为关闭连接
            print('  closing')
            mysel.unregister(connection)
            connection.close()
            # 告诉主进程停止
            keep_running = False


def accept(sock, mask):
    "有新连接的回调"
    new_connection, addr = sock.accept()    #被动接受TCP客户的连接,(阻塞式)等待连接的到来
    print('accept({})'.format(addr))
    new_connection.setblocking(False)
    mysel.register(new_connection, selectors.EVENT_READ, read)
    #对描述符进行注册，也就是对该描述符的EVENT_READ事件进行监听，当又READ事件通知时，调用回调函数read

server_address = ('localhost', 10001)
print('starting up on {} port {}'.format(*server_address))
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
server.setblocking(False)
server.bind(server_address)     #绑定(主机,端口号)到套接字
server.listen(5)                #开始TCP监听

mysel.register(server, selectors.EVENT_READ, accept)


while keep_running:
    print('waiting for I/O')
    for key, mask in mysel.select():
        callback = key.data
        callback(key.fileobj, mask)

print('shutting down')
mysel.close()


