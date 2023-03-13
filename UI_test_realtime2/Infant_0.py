# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Infant.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication


class Ui_MainWindow(object):

    def get_screen_width_and_height(self):
        self.desktop = QApplication.desktop()
        # 获取显示器分辨率大小
        self.screenRect = self.desktop.screenGeometry()
        self.screen_height = self.screenRect.height()
        self.screen_width = self.screenRect.width()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1311, 741)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMinimumSize(QtCore.QSize(561, 91))
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.widget = QtWidgets.QWidget(self.groupBox)
        self.widget.setGeometry(QtCore.QRect(10, 40, 541, 51))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.real_time_camera = QtWidgets.QPushButton(self.widget)
        self.real_time_camera.setObjectName("real_time_camera")
        self.horizontalLayout.addWidget(self.real_time_camera)
        self.btn_input = QtWidgets.QPushButton(self.widget)
        self.btn_input.setObjectName("btn_input")
        self.horizontalLayout.addWidget(self.btn_input)
        self.btn_pause = QtWidgets.QPushButton(self.widget)
        self.btn_pause.setObjectName("btn_pause")
        self.horizontalLayout.addWidget(self.btn_pause)
        self.export_log = QtWidgets.QPushButton(self.widget)
        self.export_log.setObjectName("export_log")
        self.horizontalLayout.addWidget(self.export_log)
        self.quit_system = QtWidgets.QPushButton(self.widget)
        self.quit_system.setObjectName("quit_system")
        self.horizontalLayout.addWidget(self.quit_system)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 2, 1)
        self.speech_singal_label = QtWidgets.QLabel(self.centralwidget)
        self.speech_singal_label.setMinimumSize(QtCore.QSize(681, 41))
        self.speech_singal_label.setAlignment(QtCore.Qt.AlignCenter)
        self.speech_singal_label.setObjectName("speech_singal_label")
        self.gridLayout.addWidget(self.speech_singal_label, 0, 1, 1, 2)
        self.speech_signal = QtWidgets.QGraphicsView(self.centralwidget)
        self.speech_signal.setMinimumSize(QtCore.QSize(681, 192))
        self.speech_signal.setObjectName("speech_signal")
        self.gridLayout.addWidget(self.speech_signal, 1, 1, 2, 2)
        self.label_video = QtWidgets.QLabel(self.centralwidget)
        self.label_video.setMinimumSize(QtCore.QSize(760, 531))
        self.label_video.setObjectName("label_video")
        self.gridLayout.addWidget(self.label_video, 2, 0, 3, 1)
        self.recognition_results_label = QtWidgets.QLabel(self.centralwidget)
        self.recognition_results_label.setMinimumSize(QtCore.QSize(331, 41))
        self.recognition_results_label.setAlignment(QtCore.Qt.AlignCenter)
        self.recognition_results_label.setObjectName("recognition_results_label")
        self.gridLayout.addWidget(self.recognition_results_label, 3, 1, 1, 1)
        self.statistics_results_label = QtWidgets.QLabel(self.centralwidget)
        self.statistics_results_label.setMinimumSize(QtCore.QSize(331, 41))
        self.statistics_results_label.setAlignment(QtCore.Qt.AlignCenter)
        self.statistics_results_label.setObjectName("statistics_results_label")
        self.gridLayout.addWidget(self.statistics_results_label, 3, 2, 1, 1)
        self.recognition_result = QtWidgets.QListWidget(self.centralwidget)
        self.recognition_result.setMinimumSize(QtCore.QSize(331, 311))
        self.recognition_result.setObjectName("recognition_result")
        self.gridLayout.addWidget(self.recognition_result, 4, 1, 1, 1)
        self.data_statistic_show = QtWidgets.QLabel(self.centralwidget)
        self.data_statistic_show.setMinimumSize(QtCore.QSize(331, 311))
        self.data_statistic_show.setObjectName("data_statistic_show")
        self.gridLayout.addWidget(self.data_statistic_show, 4, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1311, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.speech_singal_label.setFont(QFont('SimSun', 18))
        self.statistics_results_label.setFont(QFont('SimSun', 18))
        self.recognition_results_label.setFont(QFont('SimSun', 18))
        self.groupBox.setFont(QFont('SimSun', 18))
        self.real_time_camera.setFont(QFont('SimSun', 14))
        self.btn_input.setFont(QFont('SimSun', 14))
        self.export_log.setFont(QFont('SimSun', 14))
        self.quit_system.setFont(QFont('SimSun', 14))
        self.btn_pause.setFont(QFont('SimSun', 14))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于深度学习的婴儿监护系统"))
        self.groupBox.setTitle(_translate("MainWindow", "功能选择区"))
        self.real_time_camera.setText(_translate("MainWindow", "实时监测"))
        self.btn_input.setText(_translate("MainWindow", "加载视频"))
        self.btn_pause.setText(_translate("MainWindow", "暂停"))
        self.export_log.setText(_translate("MainWindow", "导出日志"))
        self.quit_system.setText(_translate("MainWindow", "退出系统"))
        self.speech_singal_label.setText(_translate("MainWindow", "语音信号"))
        self.recognition_results_label.setText(_translate("MainWindow", "识别结果"))
        self.statistics_results_label.setText(_translate("MainWindow", "统计结果"))
