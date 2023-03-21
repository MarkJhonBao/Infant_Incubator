import sys

from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from clip import *
import os
import cv2
from playwork import *

class MainWindow(QMainWindow, Ui_Form, cvDecode):
    def __init__(self):
        super(MainWindow, self).__init__()

        #self.input_time.clicked.connect(self.load_time)
        self.init_ui()
        self.init_work()
        self.ui.define_but.clicked.connect(self.load_time)
        self.ui.Path_but.clicked.connect(self.load_path)
        self.ui.btn_pause.clicked.connect(self.pause_video)
        self.ui.clip_but.clicked.connect(self.clip_video)
        self.ui.save_but.clicked.connect(self.save_path)



    def init_ui(self):
        self.ui = uic.loadUi("./clip.ui")

    def load_time(self):
        self.time = self.ui.input_time.text()
        self.clipvideo.cliptime = int(self.time)
        self.decodework.cliptime = int(self.time)
        print(int(self.time))



    def load_path(self):
        self.ui.btn_pause.setEnabled(True)
        #   设置文件扩展名过滤,注意用双分号间隔
        fileName, filetype = QFileDialog.getOpenFileName(self, "选取文件", "./", "Excel Files (*.mp4);;Excel Files (*.avi)")

        self.decodework.changeFlag = 1
        self.decodework.video_path = r"" + fileName
        self.playwork.playFlag = 0

        #   暂停/播放功能

    def save_path(self):

        #   设置文件扩展名过滤,注意用双分号间隔
        directory = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        print(directory)
        self.clip_video.save_path = directory





    def pause_video(self):
        if self.ui.btn_pause.text() == "暂停":
            self.ui.btn_pause.setText("播放")
            self.playwork.playFlag = 0
        else:
            self.ui.btn_pause.setText("暂停")
            self.playwork.playFlag = 1


    def init_work(self):
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        self.decodework.start()

        self.playwork = play_Work()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.ui.video_lab


        self.playwork.start()

        self.clipvideo = clip_videofun()
        self.clipvideo.clipLabel = self.ui.clip_lab


    def closeEvent(self, event):
        print("关闭线程")
        # Qt需要先退出循环才能关闭线程
        if self.decodework.isRunning():
            self.decodework.threadFlag = 0
            self.decodework.quit()
        if self.playwork.isRunning():
            self.playwork.threadFlag = 0
            self.playwork.quit()


    def clip_video(self):

        self.clipvideo.clipFlag = 1
        #self.clipvideo.threadFlag = 1
        self.clipvideo.threadFlag = 1
        self.clipvideo.start()
        self.decodework.clip_Flag =1







if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ui.show()
    sys.exit(app.exec_())
