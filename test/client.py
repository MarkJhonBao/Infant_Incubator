# Run with two terminals
import selectors
import socket

mysel = selectors.DefaultSelector()
keep_running = True

out = ""
server_address = ("localhost", 10001)                       # 连接是阻塞操作， 因此在返回之后调用setblocking()方法
print('connecting to {} port {}'.format(*server_address))

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, protocal=0)    #获取TCP/IP套接字,UDP:socket.SOCK_DGRAM
sock.connect(server_address)    #主动初始化TCP服务器连接, connect_ex():connect()函数的扩展版本,出错时返回出错码,而不是抛出异常
sock.setblocking(False)         #设置非阻塞

# 设置选择器去监听 socket 是否可写的和是否可读
mysel.register(
    sock,
    selectors.EVENT_READ | selectors.EVENT_WRITE,
)

while keep_running:
    # print('waiting for I/O')
    for key, mask in mysel.select(timeout=1):
        connection = key.fileobj
        client_address = connection.getpeername()
        # print('client({})'.format(client_address))

        if mask & selectors.EVENT_READ:
            # print('  ready to read')
            data = connection.recv(1024)
            print("if duse...")
            if data:
                # A readable client socket has data
                print('receive:  ', data)
                # bytes_received += len(data)

        if mask & selectors.EVENT_WRITE:
            # print('  ready to write')
            out = input('ready to write: ')
            # print(out)
            # print('  sending {!r}'.format(next_msg))
            sock.sendall(out.encode())
            # bytes_sent += len(next_msg)

            if out == "exit":
                print("closing...")
                keep_running = False

print('shutting down')
mysel.unregister(connection)
connection.close()
mysel.close()