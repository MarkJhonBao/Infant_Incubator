import base64
import time
import zmq
import json
import torch
import numpy as np
from network import Resnet_3D
import cv2
import pickle

if __name__ == '__main__':
    # 创建另一个套接字对象来接收数据
    context = zmq.Context()
    socket2 = context.socket(zmq.SUB)
    socket2.connect('tcp://10.2.125.73:1234')
    socket2.setsockopt(zmq.SUBSCRIBE, b'')
    print('zmq server start....')

    while 1:
        # 接收视频数据并转换为VideoClip对象
        video_data = socket2.recv()
        video_clip = pickle.loads(video_data)  # video_clip是16帧的图片列表

        print(len(video_clip))
        # model, device = load_model()
