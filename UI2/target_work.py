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
        self.xintiao_value = QLabel()
        self.tiwen_value = QLabel()
        self.xuetang_value = QLabel()
        self.huxilv_value = QLabel()
        self.wendu_value = QLabel()
        self.tiye_value = QLabel()
        self._value = QLabel()
        self.shijian_value = QLabel()
        self.xintiao_graph = QGraphicsView()
        self.tiwen_graph = QGraphicsView()
        self.xuetang_graph = QGraphicsView()
        self.huxilv_graph = QGraphicsView()
        self.wendu_graph = QGraphicsView()
        self.pingfen_graph = QGraphicsView()

    def get_targrt(self):
        xintiao = random.randint(100, 125)
        tiwen = random.randint(300, 356) / 10
        xuetang = random.randint(40, 46) / 10
        huxilv = random.randint(40, 45)
        wendu = random.randint(200, 232) / 10
        tiye = 100
        pingfen = 89
        shijian = 4

        target = [xintiao, tiwen, xuetang, huxilv, wendu, tiye, pingfen, shijian]

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
        self.xuetang_value.setText(str(xuetang) + "mm/L")
        self.huxilv_value.setText(str(huxilv) + "t/m")
        self.wendu_value.setText(str(wendu) + "℃")
        self.tiye_value.setText(str(tiye) + "mL")
        self.pingfen_value.setText(str(pingfen))
        self.shijian_value.setText(str(shijian) + "h later")

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
        self.draw_xintiao_graph()
        self.draw_tiwen_graph()
        self.draw_xuetang_graph()
        self.draw_huxilv_graph()
        self.draw_wendu_graph()
        self.draw_pingfen_graph()

    def draw_xintiao_graph(self):
        xintiao_average = 120
        xintiao_line = 0
        xintiao_sensitivity = 1

        self.xintiao_graph.scene = QGraphicsScene(self)
        self.xintiao_graph.setScene(self.xintiao_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.xintiao_graph.scene.addItem(coordinate_item)
        self.xintiao_graph.show()

        self.xintiao_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[xintiao_line][i] == 0:
                self.target_list[xintiao_line][i] = xintiao_average

        self.xintiao_graph.path.moveTo(-218, (int(-self.target_list[xintiao_line][0])
                                              + xintiao_average) * xintiao_sensitivity)

        for i in range(20):
            self.xintiao_graph.path.lineTo(QPointF(-200 + 20 * i, (int(-self.target_list[xintiao_line][i + 1])
                                                                   + xintiao_average) * xintiao_sensitivity))

        self.xintiao_graph.path.lineTo(QPointF(218, (int(-self.target_list[xintiao_line][21])
                                                     + xintiao_average) * xintiao_sensitivity))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.xintiao_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.xintiao_graph.scene.addItem(item)
        self.xintiao_graph.show()

    def draw_tiwen_graph(self):
        tiwen_average = 35
        tiwen_line = 1
        tiwen_sensitivity = 5

        self.tiwen_graph.scene = QGraphicsScene(self)
        self.tiwen_graph.setScene(self.tiwen_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.tiwen_graph.scene.addItem(coordinate_item)
        self.tiwen_graph.show()

        self.tiwen_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[tiwen_line][i] == 0:
                self.target_list[tiwen_line][i] = tiwen_average

        self.tiwen_graph.path.moveTo(-218, (int(-self.target_list[tiwen_line][0])
                                            + tiwen_average) * tiwen_sensitivity)

        for i in range(20):
            self.tiwen_graph.path.lineTo(QPointF(-200 + 20 * i, (int(-self.target_list[tiwen_line][i + 1])
                                                                 + tiwen_average) * tiwen_sensitivity))
        self.tiwen_graph.path.lineTo(QPointF(218, (int(-self.target_list[tiwen_line][21])
                                                   + tiwen_average) * tiwen_sensitivity))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.tiwen_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.tiwen_graph.scene.addItem(item)
        self.tiwen_graph.show()

    def draw_xuetang_graph(self):
        xuetang_average = 4
        xuetang_line = 2
        xuetang_sensitivity = 20

        self.xuetang_graph.scene = QGraphicsScene(self)
        self.xuetang_graph.setScene(self.xuetang_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.xuetang_graph.scene.addItem(coordinate_item)
        self.xuetang_graph.show()

        self.xuetang_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[xuetang_line][i] == 0:
                self.target_list[xuetang_line][i] = xuetang_average

        self.xuetang_graph.path.moveTo(-218, int((-self.target_list[xuetang_line][0]
                                                  + xuetang_average) * xuetang_sensitivity))
        for i in range(20):
            self.xuetang_graph.path.lineTo(QPointF(-200 + 20 * i, int((-self.target_list[xuetang_line][i + 1]
                                                                       + xuetang_average) * xuetang_sensitivity)))
        self.xuetang_graph.path.lineTo(QPointF(218, int((-self.target_list[xuetang_line][21]
                                                         + xuetang_average) * xuetang_sensitivity)))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.xuetang_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.xuetang_graph.scene.addItem(item)
        self.xuetang_graph.show()

    def draw_huxilv_graph(self):
        huxilv_average = 45
        huxilv_line = 3
        huxilv_sensitivity = 5

        self.huxilv_graph.scene = QGraphicsScene(self)
        self.huxilv_graph.setScene(self.huxilv_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.huxilv_graph.scene.addItem(coordinate_item)
        self.huxilv_graph.show()

        self.huxilv_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[huxilv_line][i] == 0:
                self.target_list[huxilv_line][i] = huxilv_average

        self.huxilv_graph.path.moveTo(-218, int((-self.target_list[huxilv_line][0]
                                                 + huxilv_average) * huxilv_sensitivity))
        for i in range(20):
            self.huxilv_graph.path.lineTo(QPointF(-200 + 20 * i, int((-self.target_list[huxilv_line][i + 1]
                                                                      + huxilv_average) * huxilv_sensitivity)))
        self.huxilv_graph.path.lineTo(QPointF(218, int((-self.target_list[huxilv_line][21]
                                                        + huxilv_average) * huxilv_sensitivity)))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.huxilv_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.huxilv_graph.scene.addItem(item)
        self.huxilv_graph.show()

    def draw_wendu_graph(self):
        wendu_average = 20
        wendu_line = 4
        wendu_sensitivity = 20

        self.wendu_graph.scene = QGraphicsScene(self)
        self.wendu_graph.setScene(self.wendu_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.wendu_graph.scene.addItem(coordinate_item)
        self.wendu_graph.show()

        self.wendu_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[wendu_line][i] == 0:
                self.target_list[wendu_line][i] = wendu_average

        self.wendu_graph.path.moveTo(-218, int((-self.target_list[wendu_line][0]
                                                + wendu_average) * wendu_sensitivity))
        for i in range(20):
            self.wendu_graph.path.lineTo(QPointF(-200 + 20 * i, int((-self.target_list[wendu_line][i + 1]
                                                                     + wendu_average) * wendu_sensitivity)))
        self.wendu_graph.path.lineTo(QPointF(218, int((-self.target_list[wendu_line][21]
                                                       + wendu_average) * wendu_sensitivity)))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.wendu_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.wendu_graph.scene.addItem(item)
        self.wendu_graph.show()

    def draw_pingfen_graph(self):
        pingfen_average = 85
        pingfen_line = 6
        pingfen_sensitivity = 5

        self.pingfen_graph.scene = QGraphicsScene(self)
        self.pingfen_graph.setScene(self.pingfen_graph.scene)

        coordinate_path = QPainterPath()
        coordinate_path.moveTo(-218, -80)
        coordinate_path.lineTo(QPointF(218, -80))
        coordinate_path.moveTo(-218, 80)
        coordinate_path.lineTo(QPointF(218, 80))
        coordinate_pen = QPen(QColor(255, 0, 0))
        coordinate_pen.setWidth(2)
        coordinate_item = QGraphicsPathItem(coordinate_path)
        coordinate_item.setPen(coordinate_pen)
        self.pingfen_graph.scene.addItem(coordinate_item)
        self.pingfen_graph.show()

        self.pingfen_graph.path = QPainterPath()
        for i in range(22):
            if self.target_list[pingfen_line][i] == 0:
                self.target_list[pingfen_line][i] = pingfen_average

        self.pingfen_graph.path.moveTo(-218, int((-self.target_list[pingfen_line][0]
                                                  + pingfen_average) * pingfen_sensitivity))
        for i in range(20):
            self.pingfen_graph.path.lineTo(QPointF(-200 + 20 * i, int((-self.target_list[pingfen_line][i + 1]
                                                                       + pingfen_average) * pingfen_sensitivity)))
        self.pingfen_graph.path.lineTo(QPointF(218, int((-self.target_list[pingfen_line][21]
                                                         + pingfen_average) * pingfen_sensitivity)))

        pen = QPen(QColor(65, 155, 255))
        pen.setWidth(2)
        item = QGraphicsPathItem(self.pingfen_graph.path)
        item.setPen(pen)
        # item.setFlag(item.ItemIsMovable)
        # item.setFlag(item.ItemIsSelectable)
        self.pingfen_graph.scene.addItem(item)
        self.pingfen_graph.show()
