import cv2
import moviepy.editor as mp
import zmq
import pickle
from queue import Queue

# 创建VideoCapture对象并读取视频文件
cap = cv2.VideoCapture(r"C:\Users\dev1se\Desktop\InfantGUI\UI2\test02.mp4")

clip_list = []

# 创建ZeroMQ上下文和套接字对象
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind('tcp://10.2.125.73:1234')
i = 0
while 1:
    # 循环读取视频帧并添加到clip列表中
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            clip_list.append(frame)
            if len(clip_list) == 16:
                # print(clip_list)
                # 将VideoClip对象转换为二进制数据
                video_data = pickle.dumps(clip_list)
                # 发送二进制数据
                socket.send(video_data)
                i += 1
                print("成功发送" + str(i))
                clip_list = []
        else:
            break

    # 释放VideoCapture对象
    cap.release()
    # # print(clip_list)
