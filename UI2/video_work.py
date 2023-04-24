import datetime
from queue import Queue
import PIL.Image as Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_agg import FigureCanvasAgg
import re

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import librosa
import os
import moviepy.editor as mp
import random
import imageio

from UI2.inference_gui import infantDetection, moderl_load, center_crop  # ,audio_model_load,audio_model_detection
from UI2.target_work import target_Work

Decode2Play = Queue()


class cvDecode(QThread):
    def __init__(self):
        super(cvDecode, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.video_path = ""  # 视频文件路径
        self.changeFlag = 0  # 判断视频文件路径是否更改
        self.cap = cv2.VideoCapture()

    def run(self):
        while self.threadFlag:
            if self.changeFlag == 1 and self.video_path != "":
                self.changeFlag = 0
                self.cap = cv2.VideoCapture(r"" + self.video_path)

            if self.video_path != "":
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    time.sleep(0.03)  # 控制读取录像的时间，连实时视频的时候改成time.sleep(0.001)，多线程的情况下最好加上，否则不同线程间容易抢占资源

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


class play_Work(QObject):
    def __init__(self):
        super(play_Work, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.playFlag = 0  # 控制播放/暂停
        self.audio_path = ""  # 音频存放的路径
        self.video_path = ""  # 视频存放的路径
        # 初始化对象
        self.playLabel = QLabel()
        self.speech_signal = QLabel()
        self.recognition_result = QListWidget()
        self.recognition_audio = QListWidget()
        self.data_statistic_show = QLabel()
        # self.xintiao_value = QLabel()
        self.head = QLabel()
        self.left_hand = QLabel()
        self.right_hand = QLabel()
        self.left_feet = QLabel()
        self.right_feet = QLabel()
        self.record_bar = QLabel()

        # 导入模型
        self.infant_detection_model = moderl_load()
        # self.aduio_detection_model = audio_model_load()

        # 初始化音频相关参数
        self.clip = []
        self.clip2 = []  # 不压缩的视频流
        self.count = 0
        self.audio_label = ["human", "device", "cry", "background"]
        self.sorted_indexes = []
        self.flag = True
        self.ret = 0
        self.cry_time = 0
        self.sum = 0
        self.start_time = 0
        self.target_Work = target_Work()
        self.act_count = [0, 0, 0, 0, 0]
        self.time1 = time.time()
        self.time2 = 0

        #   不需要重写run方法

    def play(self):
        last_result = ' '
        this_result = ' '
        self.prob_result = np.array([[0, 0, 0, 0, 0]])
        # 转换视频数据为音频数据
        # self.audio_path = self.video_to_audio(self.video_path)
        # #获取音频视频的模型
        model = self.infant_detection_model
        # audio_model=self.aduio_detection_model
        # #获取音频的数据
        # self.audio_data,self.samplate=librosa.load(self.audio_path,sr=None)
        # self.auido_duration=int(librosa.get_duration(self.audio_data,self.samplate))
        n = 0
        cnt = 0

        # self.graph_init()
        # self.target_Work.graph_init()

        while self.threadFlag:
            # print("进入",switchflag)
            if not Decode2Play.empty():
                # 读取队列中的数据并处理
                frame = Decode2Play.get()
                tmp_ = cv2.resize(frame, (480, 270))
                # tmp_ = cv2.resize(frame, (192, 108))
                tmp_ = cv2.cvtColor(tmp_, cv2.COLOR_BGR2RGB)
                self.clip2.append(tmp_)

                if cnt % 5 == 0:
                    cnt = 0
                    tmp_ = center_crop(cv2.resize(frame, (171, 128)))
                    tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                    self.clip.append(tmp)

                cnt += 1

                # if self.count>self.auido_duration*2:
                #     self.count=0
                #     self.sorted_indexes=[]
                #     self.flag = True
                #     self.ret = 0
                #     self.cry_time=0
                #     self.sum=0
                #     self.start_time = 0

                # if cnt%12==0 and self.playFlag ==1:
                # self.audio_clip = audio_model_detection(audio_model, self.audio_data[self.count * self.samplate // 2:(self.count + 1) * self.samplate // 2]
                #                                         , self.samplate)
                # print(self.audio_clip)
                # self.show_audio(self.audio_clip, self.count)
                # self.count += 1
                # imagea=self.draw_audio(self.audio_clip).toqpixmap()
                # self.speech_signal.setPixmap(imagea)
                # self.speech_signal.setScaledContents(True)

                if len(self.clip) == 16 and self.playFlag == 1:
                    frame, class_result, prob_result = infantDetection(model, clip=self.clip, frame=frame)
                    self.prob_result = prob_result
                    c = sum(sum(class_result))
                    # print(sum(class_result))
                    result = sum(class_result)
                    class_result = str(class_result)
                    # print(c)

                    # n += 1
                    # self.png_to_gif(n)
                    # todo cv png转gif
                    if c >= 3:
                        n += 1
                        self.png_to_gif(n)

                    this_result = class_result
                    self.clip = []
                    self.clip2 = []

                    # self.target_Work.add_target()
                    self.target_Work.show_target_new()
                    self.target_Work.draw_graph()

                    self.save_record(result)
                    self.show_record()

                if self.playFlag == 1:
                    frame = cv2.resize(frame, (340, 190), cv2.INTER_LINEAR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    qimg = QImage(frame.data, frame.shape[1], frame.shape[0],
                                  QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
                    self.playLabel.setPixmap(QPixmap.fromImage(qimg))  # 图像在QLabel上展示
                    if (last_result != this_result):
                        now = datetime.datetime.now()
                        # print(self.show_data(prob_result))
                        self.recognition_result.addItem(
                            now.strftime("%m-%d %H:%M:%S") + '识别结果：' + self.show_data(self.prob_result))
                        # self.data_statistic(this_result)  # 统计结果自增1
                        piximage_data_statistic = self.draw_bar().toqpixmap()
                        self.data_statistic_show.setPixmap(piximage_data_statistic)
                        self.data_statistic_show.setScaledContents(True)
                    last_result = this_result
                # cv2.putText(frame, class_result, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
            # time.sleep(0.001)

    def show_data(self, prob_result):
        probs = []
        res = str
        name_list = ['头部', '左手', '左腿', '右手', '右腿']
        for i in range(len(prob_result)):
            if prob_result[i] > 0.5:
                prob_result[i] = prob_result[i]
                probs.append(name_list[i] + '(' + str("%.2f" % (prob_result[i] * 100)) + '%)')
        if len(probs):
            res = ('+'.join(probs))
        else:
            res = "正常"
        return res

    def show_audio(self, data, count):
        if len(self.sorted_indexes) >= 3:
            array1 = np.array(self.sorted_indexes)
            while True:
                # print(self.ret,count)
                if self.ret >= count:
                    break;
                # print(is_cry(sorted_indexes,i,length),flag,i)
                if self.is_cry(array1, self.ret) == True:
                    self.ret = self.ret + 1
                    if self.flag == True:
                        self.flag = False
                        self.cry_time += 1
                        # print("第 {} 次哭：开始时间:{}s".format(self.cry_time, (ret-1) * 0.5))
                        now = datetime.datetime.now()
                        self.recognition_audio.addItem(now.strftime("%m-%d %H:%M:%S") + "第"
                                                       + str(self.cry_time) + "次哭：开始时间：" + str(
                            (self.ret - 1) * 0.5) + "s")
                        self.start_time = (self.ret - 1) * 0.5
                        print(self.start_time)
                    # print("start:",i, cry_time)
                elif self.is_cry(array1, self.ret) == False:
                    self.ret = self.ret + 1
                    if self.flag == False:
                        self.flag = True
                        # print(" 结束时间:{}s".format((ret - 1) * 0.5))
                        self.recognition_audio.addItem(" 结束时间：" + str((self.ret - 1) * 0.5) + "s")
                        self.sum = self.sum + (self.ret - 1) * 0.5 - self.start_time
                        print(self.sum, self.start_time)
            if count >= self.auido_duration * 2:
                if self.flag == False:
                    self.recognition_audio.addItem(" 结束时间：" + str((self.ret) * 0.5) + "s")
                self.recognition_audio.addItem("哭声总时长：" + str(self.sum) + "s")
            self.sorted_indexes.append(np.argsort(data, axis=-1)[-1: -4 - 1: -1])
        else:
            self.sorted_indexes.append(np.argsort(data, axis=-1)[-1: -4 - 1: -1])

    # 判断每帧是否哭
    def is_cry(self, sorted_list, index):
        length = len(sorted_list[:, 0])
        if index == 0:
            index += 1
        if index == length - 1:
            index -= 1
        if index > 0 and index < length - 1:
            # print(self.audio_label[sorted_list[index, 0]])
            if self.audio_label[sorted_list[index, 0]] == "cry":
                if self.audio_label[sorted_list[index - 1, 0]] == "cry" or self.audio_label[
                    sorted_list[index + 1, 0]] == "cry":
                    return True
                else:
                    return False
            else:
                if self.audio_label[sorted_list[index - 1, 0]] == "cry" and self.audio_label[
                    sorted_list[index + 1, 0]] == "cry":
                    return True
                else:
                    return False

    # 显示概率
    def draw_bar(self):
        # print("start")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        name_list = ['头部', '左手', '左腿', '右手', '右腿']
        # num_list = self.prob_result
        num_list = [1, 1, 1, 1, 1]
        fig = plt.figure(figsize=(6.5, 4))

        plt.ylim(0, 1)
        plt.bar(range(len(num_list)), num_list, tick_label=name_list)
        # print("end")
        canvas = FigureCanvasAgg(plt.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image

    def draw_audio(self, data):
        # print("start")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        labels = ["人说话声", "设备声", "哭声", "背景噪声"]
        fig = plt.figure(figsize=(6.5, 4))
        # print("end")
        plt.ylim(0, 1)
        plt.bar(range(len(data)), data, tick_label=labels)

        canvas = FigureCanvasAgg(plt.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image

    # 转化视频数据为音频数据,并返回音频的存放位置
    def video_to_audio(self, video_path):
        # print("start")
        output_path = os.path.join(
            r"C:\Users\dev1se\Desktop\InfantGUI\audio\output_data\\" + video_path.split("/")[-1][:-4] + ".wav")
        my_clip = mp.VideoFileClip(video_path)
        my_clip.audio.write_audiofile(output_path)
        global switchflag
        switchflag = 1
        # print("end")
        return output_path

    def png_to_gif(self, n):
        # img_lst = os.listdir(path)
        # frames = []
        frame = self.clip2
        path = "C:\\Users\\dev1se\\Desktop\\InfantGUI\\gif\\save" + str(n) + ".gif"
        imageio.mimsave(path, frame, 'GIF', fps=16)

    def save_record(self, result):
        # current_time = datetime.datetime.now()
        # print("current_time:    " + str(current_time))
        name_list = ['头部', '左手', '左腿', '右手', '右腿']
        record_list = []
        for i in range(5):
            if result[i] == 1:
                self.act_count[i] += 1
        record_list.append(name_list[i])
        # record = "".join(record_list)
        self.time2 = time.time()
        if (self.time2 - self.time1) >= 10:
            self.time1 = self.time2
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            # with open(r"C:\\Users\\dev1se\\Desktop\\InfantGUI\\日志.txt", "a", encoding='utf-8') as f:
            #     f.write(current_time+"    "+"过去30秒内 头部运动"+self.act_count[0].__str__()+"次 左手运动"+self.act_count[1].__str__() +
            #             "次 左腿运动"+self.act_count[2].__str__()+"次 右手运动"+self.act_count[3].__str__()+"次 右腿运动"+self.act_count[4].__str__()+"次"+"\n")
            with open(r"日志.txt", "a", encoding='utf-8') as f:
                f.write(current_time + "    " + self.act_count.__str__() + "\n")
            self.act_count = [0, 0, 0, 0, 0]

    def read_record(self):
        record_list = []
        with open(r"日志.txt", "r") as f:
            lines = f.readlines()
            last_lines = lines[-10:]
            for line in last_lines:
                matches = re.findall(r'(\d+)[,\]]', line)
                int_list = []
                for i in matches:
                    int_list.append(int(i))
                record_list.append(int_list)
        act_sum = [sum(i) for i in zip(*record_list)]
        return act_sum

    def show_record(self):
        act_sum = self.read_record()
        self.head.setText("头部运动：" + str(act_sum[0]) + "次")
        self.left_hand.setText("左手运动：" + str(act_sum[1]) + "次")
        self.right_hand.setText("右手运动：" + str(act_sum[2]) + "次")
        self.left_feet.setText("左脚运动：" + str(act_sum[3]) + "次")
        self.right_feet.setText("右脚运动：" + str(act_sum[4]) + "次")

        record_bar = self.draw_record_bar().toqpixmap()
        self.record_bar.setPixmap(record_bar)
        self.record_bar.setScaledContents(True)

    def draw_record_bar(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        name_list = ['头部', '左手', '右手', '左腿', '右腿']
        # num_list = self.prob_result
        num_list = self.read_record()
        fig = plt.figure(figsize=(510 / 110, 260 / 110), dpi=110)

        # plt.ylim(0, 1)
        plt.bar(range(len(num_list)), num_list, tick_label=name_list)
        # print("end")
        canvas = FigureCanvasAgg(plt.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image
