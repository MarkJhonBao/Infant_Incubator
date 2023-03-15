
import datetime
import time
from queue import Queue
import os
from PyQt5.QtCore import QThread
import  cv2
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *

Decode2Play = Queue()

#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

class cvDecode(QThread):
    def __init__(self):
        super(cvDecode, self).__init__()
        self.threadFlag = 0     #   控制线程退出
        self.video_path = ""    #   视频文件路径
        self.changeFlag = 0     #   判断视频文件路径是否更改
        self.cap = cv2.VideoCapture()

    def run(self):
        while self.threadFlag:
            if self.changeFlag == 1 and self.video_path !="":
                self.changeFlag = 0
                self.cap = cv2.VideoCapture(r""+self.video_path)

            if self.video_path !="":
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    time.sleep(0.04)   # 控制读取录像的时间，连实时视频的时候改成time.sleep(0.001)，多线程的情况下最好加上，否则不同线程间容易抢占资源

                    # 下面两行代码用来控制循环播放的，如果不需要可以删除
                    if frame is None:
                        self.cap = cv2.VideoCapture(r"" + self.video_path)

                    if ret:
                        Decode2Play.put(frame)  # 解码后的数据放到队列中
                    del frame  # 释放资源
                else:
                    #   控制重连
                    self.cap = cv2.VideoCapture(r"" + self.video_path)
                    time.sleep(0.01)



class play_Work(QThread):
    def __init__(self):
        super(play_Work, self).__init__()
        self.threadFlag = 0         #   控制线程退出
        self.playFlag = 0           #   控制播放/暂停
        self.playLabel = QLabel()   #   初始化QLabel对象\
        self.clipLabel = QLabel()
        self.clipFlag = 0
        self.fps = 24
        self.cliptime = 0
        self.i = 0
        self.save_path =''
    def run(self):
        while self.threadFlag:
            if not Decode2Play.empty():
                frame = Decode2Play.get()
                if self.playFlag == 1:

                    self.frame = frame
                    frame = cv2.resize(frame, (760, 480), cv2.INTER_LINEAR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    qimg = QImage(frame.data, frame.shape[1], frame.shape[0],
                                  QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
                    self.playLabel.setPixmap(QPixmap.fromImage(qimg))   #   图像在QLabel上展示

                    if self.clipFlag:
                        if self.i==0:
                            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                            #video = cv2.VideoWriter(r"D:\hdu\clip.mp4", fourcc, self.fps, (self.frame.shape[1], self.frame.shape[0]))
                            video = cv2.VideoWriter(self.save_path+'/clip1.mp4', fourcc, self.fps, (self.frame.shape[1], self.frame.shape[0]))
                            print(self.fps)
                        if self.i < (self.fps * self.cliptime):
                            print(self.i)
                            video.write(self.frame)
                            self.i = self.i+1
                        else:
                            self.clipFlag = 0
                            self.i = 0
                            video.release()
                            self.clipLabel.setText("保存成功")

                        #self.clipLabel.setPixmap(QPixmap.fromImage(qimg))


            time.sleep(0.001)



