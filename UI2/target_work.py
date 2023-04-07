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

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import librosa
import os
import moviepy.editor as mp
import random
import imageio
from UI2.inference_gui import infantDetection, moderl_load, center_crop  # ,audio_model_load,audio_model_detection


class target_Work(QObject):
    def __init__(self):
        super(target_Work, self).__init__()
        self.target_list = np.zeros((8, 22), dtype=float)
        self.count = 0
        self.xintiao_value = QLabel()
        self.tiwen_value = QLabel()
        self.xuetang_value = QLabel()
        self.huxilv_value = QLabel()
        self.wendu_value = QLabel()
        self.tiye_value = QLabel()
        self._value = QLabel()
        self.shijian_value = QLabel()
        self.xintiao_graph = QLabel()
        self.tiwen_graph = QLabel()
        self.xuetang_graph = QLabel()
        self.huxilv_graph = QLabel()
        self.wendu_graph = QLabel()
        self.pingfen_graph = QLabel()

    def get_targrt(self):
        # xintiao = random.randint(120, 123)
        # tiwen = random.randint(359, 362) / 10
        # xuetang = random.randint(45, 46) / 10
        # huxilv = random.randint(58, 60)
        # wendu = random.randint(230, 232) / 10
        # tiye = 100
        # pingfen = 89
        # shijian = 4

        with open(r"xintiao.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        xintiao = data[self.count+21].__int__()

        with open(r"tiwen.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        tiwen = round(data[self.count+21], 1)

        with open(r"xuetang.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        xuetang = round(data[self.count+21], 1)

        with open(r"huxilv.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        huxilv = data[self.count+21].__int__()

        with open(r"wendu.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        wendu = round(data[self.count+21], 1)

        with open(r"pingfen.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        pingfen = data[self.count+21].__int__()

        target = [xintiao, tiwen, xuetang, huxilv, wendu,  pingfen]

        return target

    def add_target(self):
        new_target = self.get_targrt()
        for i in range(8):
            for j in range(21):
                self.target_list[i][j] = self.target_list[i][j + 1]
                # print(self.target_list[i][j])
        for i in range(8):
            self.target_list[i][21] = new_target[i]

    def show_target(self):
        target = []
        for i in range(8):
            target.append(self.target_list[i][21])

        xintiao = target[0]
        tiwen = target[1]
        xuetang = target[2]
        huxilv = target[3]
        wendu = target[4]
        tiye = target[5]
        pingfen = target[6]
        shijian = target[7]

        self.xintiao_value.setText(str(xintiao) + "t/m")
        self.tiwen_value.setText(str(tiwen) + "℃")
        self.xuetang_value.setText(str(xuetang) + "mmol/L")
        self.huxilv_value.setText(str(huxilv) + "t/m")
        self.wendu_value.setText(str(wendu) + "℃")
        self.tiye_value.setText(str(tiye) + "mL")
        self.pingfen_value.setText(str(pingfen))
        self.shijian_value.setText(str(shijian) + "h later")

    def show_target_new(self):
        target = self.get_targrt()

        xintiao = target[0]
        tiwen = target[1]
        xuetang = target[2]
        huxilv = target[3]
        wendu = target[4]
        pingfen = target[5]

        self.xintiao_value.setText(str(xintiao) + "t/m")
        self.tiwen_value.setText(str(tiwen) + "℃")
        self.xuetang_value.setText(str(xuetang) + "mmol/L")
        self.huxilv_value.setText(str(huxilv) + "t/m")
        self.wendu_value.setText(str(wendu) + "℃")
        self.pingfen_value.setText(str(pingfen))

    def graph_init(self):
        self.xintiao_graph.scene = QGraphicsScene(self)
        self.xintiao_graph.setScene(self.xintiao_graph.scene)
        self.xintiao_graph.path = QPainterPath()
        self.xintiao_graph.path.moveTo(-218, -80)
        self.xintiao_graph.path.lineTo(QPointF(218, -80))
        self.xintiao_graph.path.moveTo(-218, 80)
        self.xintiao_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.xintiao_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.xintiao_graph.scene.addItem(item)
        self.xintiao_graph.show()

        self.tiwen_graph.scene = QGraphicsScene(self)
        self.tiwen_graph.setScene(self.tiwen_graph.scene)
        self.tiwen_graph.path = QPainterPath()
        self.tiwen_graph.path.moveTo(-218, -80)
        self.tiwen_graph.path.lineTo(QPointF(218, -80))
        self.tiwen_graph.path.moveTo(-218, 80)
        self.tiwen_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.tiwen_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.tiwen_graph.scene.addItem(item)
        self.tiwen_graph.show()

        self.xuetang_graph.scene = QGraphicsScene(self)
        self.xuetang_graph.setScene(self.xuetang_graph.scene)
        self.xuetang_graph.path = QPainterPath()
        self.xuetang_graph.path.moveTo(-218, -80)
        self.xuetang_graph.path.lineTo(QPointF(218, -80))
        self.xuetang_graph.path.moveTo(-218, 80)
        self.xuetang_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.xuetang_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.xuetang_graph.scene.addItem(item)
        self.xuetang_graph.show()

        self.wendu_graph.scene = QGraphicsScene(self)
        self.wendu_graph.setScene(self.wendu_graph.scene)
        self.wendu_graph.path = QPainterPath()
        self.wendu_graph.path.moveTo(-218, -80)
        self.wendu_graph.path.lineTo(QPointF(218, -80))
        self.wendu_graph.path.moveTo(-218, 80)
        self.wendu_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.wendu_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.wendu_graph.scene.addItem(item)
        self.wendu_graph.show()

        self.huxilv_graph.scene = QGraphicsScene(self)
        self.huxilv_graph.setScene(self.huxilv_graph.scene)
        self.huxilv_graph.path = QPainterPath()
        self.huxilv_graph.path.moveTo(-218, -80)
        self.huxilv_graph.path.lineTo(QPointF(218, -80))
        self.huxilv_graph.path.moveTo(-218, 80)
        self.huxilv_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.huxilv_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.huxilv_graph.scene.addItem(item)
        self.huxilv_graph.show()

        self.pingfen_graph.scene = QGraphicsScene(self)
        self.pingfen_graph.setScene(self.pingfen_graph.scene)
        self.pingfen_graph.path = QPainterPath()
        self.pingfen_graph.path.moveTo(-218, -80)
        self.pingfen_graph.path.lineTo(QPointF(218, -80))
        self.pingfen_graph.path.moveTo(-218, 80)
        self.pingfen_graph.path.lineTo(QPointF(218, 80))
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.pingfen_graph.path)
        item.setPen(pen)
        item.setFlag(item.ItemIsMovable)
        item.setFlag(item.ItemIsSelectable)
        self.pingfen_graph.scene.addItem(item)
        self.pingfen_graph.show()

    def draw_graph(self):
        self.count += 1

        xintiao = self.draw_xintiao().toqpixmap()
        self.xintiao_graph.setPixmap(xintiao)
        self.xintiao_graph.setScaledContents(True)

        tiwen = self.draw_tiwen().toqpixmap()
        self.tiwen_graph.setPixmap(tiwen)
        self.tiwen_graph.setScaledContents(True)

        xuetang = self.draw_xuetang().toqpixmap()
        self.xuetang_graph.setPixmap(xuetang)
        self.xuetang_graph.setScaledContents(True)

        huxilv = self.draw_huxilv().toqpixmap()
        self.huxilv_graph.setPixmap(huxilv)
        self.huxilv_graph.setScaledContents(True)

        wendu = self.draw_wendu().toqpixmap()
        self.wendu_graph.setPixmap(wendu)
        self.wendu_graph.setScaledContents(True)

        pingfen = self.draw_pingfen().toqpixmap()
        self.pingfen_graph.setPixmap(pingfen)
        self.pingfen_graph.setScaledContents(True)

    def draw_xintiao(self):
        # print("start")
        plt.rcParams['font.family'] = ['SimHei']
        # plt.figure(figsize=(5, 3))
        with open(r"xintiao.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        # 绘制折线图
        sub_numbers = data[self.count:self.count+21]
        plt.title(r'温度时间统计', fontsize=5)
        # plt.xlabel('Time')
        # plt.ylabel('Temperature')
        # plt.ylim(27, 40)
        fig = plt.figure(figsize=(310 / 90, 220 / 90), dpi=90)

        plt.plot(sub_numbers, color='#f89588', linewidth=2, linestyle='-', label='心率')
        plt.yticks([120, 121, 122, 123, 124])
        plt.grid()
        plt.legend()
        # plt.show()
        # print("end")
        canvas = FigureCanvasAgg(plt.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image

    def draw_tiwen(self):
        # print("start")
        plt.rcParams['font.family'] = ['SimHei']
        # plt.figure(figsize=(5, 3))
        with open(r"tiwen.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        # 绘制折线图
        sub_numbers = data[self.count:self.count + 21]
        # plt.title(r'温度时间统计', fontsize=5)
        # plt.xlabel('Time')
        # plt.ylabel('Temperature')
        # plt.ylim(120, 130)

        fig = plt.figure(figsize=(310 / 90, 220 / 90), dpi=90)

        plt.plot(sub_numbers, color='#63b2ee', linewidth=2, linestyle='-', label='体温')
        plt.yticks([35.0, 35.5, 36.0, 36.5, 37.0])
        plt.grid()
        plt.legend()
        # plt.show()
        # print("end")
        canvas = FigureCanvasAgg(plt.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image

    def draw_xuetang(self):
        # print("start")
        plt.rcParams['font.family'] = ['SimHei']
        # plt.figure(figsize=(5, 3))
        with open(r"xuetang.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        # 绘制折线图
        sub_numbers = data[self.count:self.count + 21]
        plt.title(r'温度时间统计', fontsize=5)
        # plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.ylim(27, 40)
        fig = plt.figure(figsize=(310 / 90, 220 / 90), dpi=90)

        plt.plot(sub_numbers, color='#76da91', linewidth=2, linestyle='-', label='血糖')
        plt.yticks([3, 4, 5, 6, 7])
        plt.grid()
        plt.legend()
        # plt.show()
        # print("end")
        canvas = FigureCanvasAgg(plt.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image

    def draw_huxilv(self):
        # print("start")
        plt.rcParams['font.family'] = ['SimHei']
        # plt.figure(figsize=(5, 3))
        with open(r"huxilv.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        # 绘制折线图
        sub_numbers = data[self.count:self.count + 21]
        plt.title(r'温度时间统计', fontsize=5)
        # plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.ylim(27, 40)
        fig = plt.figure(figsize=(310 / 90, 220 / 90), dpi=90)

        plt.plot(sub_numbers, color='#f8cb7f', linewidth=2, linestyle='-', label='呼吸率')
        plt.yticks([55, 56, 57, 58, 59])
        plt.grid()
        plt.legend()
        # plt.show()
        # print("end")
        canvas = FigureCanvasAgg(plt.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image

    def draw_wendu(self):
        # print("start")
        plt.rcParams['font.family'] = ['SimHei']
        # plt.figure(figsize=(5, 3))
        with open(r"wendu.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        # 绘制折线图
        sub_numbers = data[self.count:self.count + 21]
        plt.title(r'温度时间统计', fontsize=5)
        # plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.ylim(27, 40)
        fig = plt.figure(figsize=(310 / 90, 220 / 90), dpi=90)

        plt.plot(sub_numbers, color='#7cd6cf', linewidth=2, linestyle='-', label='温度')
        plt.yticks([22.0, 22.5, 23.0, 23.5, 24])
        plt.grid()
        plt.legend()
        # plt.show()
        # print("end")
        canvas = FigureCanvasAgg(plt.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image

    def draw_pingfen(self):
        # print("start")
        plt.rcParams['font.family'] = ['SimHei']
        # plt.figure(figsize=(5, 3))
        with open(r"pingfen.txt", "r") as f:
            data = [float(line.strip()) for line in f]
        # 绘制折线图
        sub_numbers = data[self.count:self.count + 21]
        plt.title(r'温度时间统计', fontsize=5)
        # plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.ylim(27, 40)
        fig = plt.figure(figsize=(310 / 90, 220 / 90), dpi=90)

        plt.plot(sub_numbers, color='#3232CD', linewidth=2, linestyle='-', label='评分')
        plt.yticks([75, 80, 85, 90, 95])
        plt.grid()
        plt.legend()
        # plt.show()
        # print("end")
        canvas = FigureCanvasAgg(plt.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image
