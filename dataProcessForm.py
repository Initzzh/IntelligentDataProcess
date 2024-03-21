import sys
from PyQt5.QtWidgets import QWidget,QAbstractItemView,QMessageBox,QFileDialog,QHeaderView
from PyQt5.QtGui import QStandardItemModel,QStandardItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import pyqtgraph.exporters

import matplotlib.pyplot as plt
import time
import datetime


# from FormUI import Ui_Form
# from FormUI import Ui_Form
from FormUI import Ui_Form
# from process import ProcessCurve
from process import ProcessCurve

import numpy as np
import pyqtgraph as pg



from scipy import signal
import pandas as pd

from PyQt5 import QtCore
import qdarkstyle
    
from qdarkstyle.light.palette import LightPalette

from qdarkstyle import *
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)



class MyForm(QWidget,Ui_Form):
    # # 曲线文件路径
    # filePathSignal = QtCore.pyqtSignal(str)
    # 创建数据传输信号， 用来传输数据
    # refresh_analyseData_signal = QtCore.pyqtSignal(dict)
    def __init__(self):
        # 添加一个线区域选择项目，起始区间在B到C
        super(MyForm,self).__init__()
        self.lr = pg.LinearRegionItem(movable=False)
        self.lr_A = pg.LinearRegionItem( movable=False)

        self.lr_fit_sample = pg.LinearRegionItem( movable=False)
        

        # 是否为第一次绘制计算结果，用来判断使用p.plot()还是p.setData()
        self.isPlotResult=False
        self.isChangedPoint=False
        self.isSmooth = False

        self.setupUi(self)
        self.init_disable(False, 1, 1)
        

        #限制只能输入数字
        from PyQt5.QtGui import QRegExpValidator, QFont
        from PyQt5.QtCore import QRegExp
        
        reg = QRegExp('[0-9]+$')
        validator = QRegExpValidator(reg, self)
        self.textEditA.setValidator(validator)
        self.textEditB.setValidator(validator)
        self.textEditC.setValidator(validator)
        self.textEditProcessLeft.setValidator(validator)
        self.textEditProcessRight.setValidator(validator)
        self.textEditParameter1.setValidator(validator)
        # self.textEditParameter2.setValidator(validator)
        # self.textEditParameter3.setValidator(validator)
        self.textEdit_FitCurve.setValidator(validator)

         #一些初始化参数，曲线数据处理区间
        self.params = [1001, 20000]
        # 进入页面的初始状态
        self.flag1 = 0
        self.flagfit=0

        self.processCurve  = ProcessCurve()
        
        self.p1, self.p2, self.p3,self.p4, self.p5, self.p6, self.p7 = self.set_graph_ui()  # 设置绘图窗口

        # 绘制表格
        # self.tableView.addItem()
        self.tableModel = QStandardItemModel(11,5)
        self.tableModel.setHorizontalHeaderLabels(['位移(mm)','抗力(kN)','分段刚度(kN/m)','上宽(kN)','下宽(kN)'])

        self.tableView.setModel(self.tableModel)
        #隐藏行号
        header=self.tableView.verticalHeader()
        header.hide()

        # 设置单元格列宽
        # self.tableWidget.setColumnWidth(0, 40)
        # 固定列宽大小，用户不可编辑列宽
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableView.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents)
        self.tableView.horizontalHeader().setStyleSheet(
            "QHeaderView::section{background-color:rgb(155, 194, 230);font:7.8pt '宋体';color: black;};")
        # 设置行宽

        for i in range(11):
            self.tableView.setRowHeight(i,21.5) 
        # 滚动条隐藏
        self.tableView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  
        self.tableView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
         
        self.tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)

         # 滑动条B的值变化
        self.SliderB.valueChanged.connect(self.SliderValueChangedSlot_B)
        # 滑动条A的值变化
        self.SliderA.valueChanged.connect(self.SliderValueChangedSlot_A)
        # 滑动条C的值变化
        self.SliderC.valueChanged.connect(self.SliderValueChangedSlot_C)

        # 滑动条Amp变化
        self.SliderAmp.valueChanged.connect(self.SliderValueChangedSlot_Amp)

        # 滑动拟合sample_start滑动条
        self.sample_start_Slider.valueChanged.connect(self.SliderValueChangedSlot_SampleStart)

        self.sample_end_Slider.valueChanged.connect(self.SliderValueChangedSlot_SampleStart)

        # self.lineEdit_fit_sample_end.setText(str(fit_sample_end)))
        # 自动选择A,B,C三点的方法
        self.mathButton.clicked.connect(self.click_mathButton)
        self.IntelligentButton.clicked.connect(self.click_IntelligentButton)

        # A,B,C三点make
        self.ButtonMake.clicked.connect(self.click_buttonMake)

        self.ButtonBLeft.clicked.connect(self.click_ButtonBLeft)
        self.ButtonCLeft.clicked.connect(self.click_ButtonCLeft)
        self.ButtonBRight.clicked.connect(self.click_ButtonBRight)
        self.ButtonCRight.clicked.connect(self.click_ButtonCRight)
        self.ButtonA.clicked.connect(self.click_ButtonA)
        self.ButtonB.clicked.connect(self.click_ButtonB)
        self.ButtonC.clicked.connect(self.click_ButtonC)
        # self.ButtonFitCurve.clicked.connect(self.click_ButtonFItCurve)

        #改变1001-12999
        self.Button_Change_Region.clicked.connect(self.click_Button_Change_Region)#

        #Reset
        self.ButtonReset.clicked.connect(self.click_Button_Reset)#

        #ParametersButton
        self.parametersMakeButton.clicked.connect(self.click_parametersMakeButton)

        #ButtonFitCurve
        # self.ButtonFitCurve.clicked.connect(self.click_ButtonFItCurve)
        self.ButtonFitCurve.clicked.connect(self.click_Button_ButtonFitCurve)

        self.fileButton.clicked.connect(self.click_fileSelectButton)
        self.saveOptionButton.clicked.connect(self.click_saveOptionButton)
        self.saveMakeButton.clicked.connect(self.click_saveMakeButton)

        # 设置窗口风格
        # self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        # self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5',Palette=LightPalette()))
        self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5',palette=LightPalette()))
        # self.refresh_analyseData_signal.connect(self.processInit)

    # 从主窗口传递实验文件路径
    def set_file(self,filePath):
        self.file=filePath


    def click_mathButton(self):
        # 数学方法确定A,B,C三点
        self.processCurve.determine_a_b_c()

        self.lr.setRegion([self.processCurve.pointB-1, self.processCurve.pointC])
        self.lr_A.setRegion([self.processCurve.pointA, self.processCurve.pointA+2])

         ####设置滑动条B
        self.SliderA.setValue(self.processCurve.pointA)
        
        ####设置滑动条B
        self.SliderB.setValue(self.processCurve.pointB)

        ####设置滑动条C
        self.SliderC.setValue(self.processCurve.pointC)

        self.textEditA.setText(str(self.processCurve.pointA))
        self.textEditB.setText(str(self.processCurve.pointB))
        self.textEditC.setText(str(self.processCurve.pointC))




    def click_IntelligentButton(self):

        # 模型预测方法确定A,B,C三点
        self.processCurve.model_predict_a_b_c()

        self.lr.setRegion([self.processCurve.pointB-1, self.processCurve.pointC])
        self.lr_A.setRegion([self.processCurve.pointA, self.processCurve.pointA+2])

         ####设置滑动条B
        self.SliderA.setValue(self.processCurve.pointA)
        
        ####设置滑动条B
        self.SliderB.setValue(self.processCurve.pointB)

        ####设置滑动条C
        self.SliderC.setValue(self.processCurve.pointC)

        self.textEditA.setText(str(self.processCurve.pointA))
        self.textEditB.setText(str(self.processCurve.pointB))
        self.textEditC.setText(str(self.processCurve.pointC))

    def click_Button_ButtonFitCurve(self):
        fit_sample_start = int(self.lineEdit_sample_start.text())
        fit_sample_end = int(self.lineEdit_sample_end.text())
        fit_degree = int(self.textEdit_FitCurve.text())
        # L = self.processCurve.pointB
        # M = int(self.textEdit_FitCurve.text())
        L = self.processCurve.pointB
        fit_sample_start = int(self.lineEdit_sample_start.text()) # 拟合采样开始点
        fit_sample_end = int(self.lineEdit_sample_end.text()) # 拟合采样结束点 ，拟合开始位置
        fit_end= self.processCurve.pointC  # 拟合结束位置
        # 根据平滑后的数据拟合
        X = self.xx[fit_sample_start:fit_sample_end]
        Y = self.processCurve.dfSmooth[fit_sample_start-self.processCurve.pointB:fit_sample_end-self.processCurve.pointB]
        # Y=  self.process
        # Y = self.process
        # X = self.xx[L:M]
        # Y = self.yy[L:M]
        coef = np.polyfit(X, Y, fit_degree) # 计算拟合曲线
        y_fit = np.poly1d(coef)
        # y_fit = np.polyval()

        y_f = y_fit(self.xx[fit_sample_end-1:fit_end])
        # y_f = y_fit(self.processCurve.dfSmooth[fit_sample_end-self.processCurve.pointB:])
        region_x = self.xx[self.processCurve.pointB-1:self.processCurve.pointC]
        region_y = self.yy.copy()
        # region_y[M:R] = y_f
        self.processCurve.dfSmooth[fit_sample_end-self.processCurve.pointB:] = y_f.copy()
        # self.processCurve.df['data'][M:R] = y_f.copy()
        region_y = region_y[self.processCurve.pointB-1:fit_end]
        
        # self.smoothBCLine.setData(region_x, region_y)
        self.smoothBCLine.setData(self.xx[self.processCurve.pointB-1:self.processCurve.pointC],self.processCurve.dfSmooth)
        self.line2.setData(region_x, self.yy[self.processCurve.pointB-1:fit_end])


    #拟合
    def click_ButtonFItCurve(self):
        # self.textEdit_FitCurve.text()
        self.processCurve.fit_tail(int(self.textEdit_FitCurve.text()))
        fitData=np.array(self.processCurve.fitdf.data)
        # self.line1= self.p1.plot(self.xx[8400:9400],fitData[8400:9400],pen=pg.mkPen('r',width=2))
        if self.flagfit==0:
            self.line1= self.p1.plot(self.xx,fitData,pen=pg.mkPen('r',width=2))
            self.flagfit=1
        else:
            self.line1.setData(self.xx, fitData)


    def click_Button_Reset(self):
        self.processCurve.set_df(self.file)
        #确定A,B,C点
        self.processCurve.determine_a_b_c()

        # self.df = pd.read_csv(file,header=None) 
        # self.df['time'] = self.df[0].str.split()
        # self.df['tick'] = self.df['time'].map(lambda x:eval(x[0]))
        # self.df['data'] = self.df['time'].map(lambda x:eval(x[1]))
        # self.df.pop('time')
        
        # df_origin = self.df.copy()
  
        self.xx = np.array(self.processCurve.df['data'].index)
        self.yy = np.array(self.processCurve.df['data'])
        df_max = max(self.processCurve.df['data'])
        df_min = min(self.processCurve.df['data'])


        #绘图
        #self.p1.clear()
        self.line0.setData(self.xx, self.yy)
        
        # 添加一个线区域选择项目，起始区间在B到C
        #self.lr = pg.LinearRegionItem([self.processCurve.pointB-1, self.processCurve.pointC])
        #self.lr_A = pg.LinearRegionItem([self.processCurve.pointA, self.processCurve.pointA+2], movable=False)
        self.lr.setRegion([self.processCurve.pointB-1, self.processCurve.pointC])
        self.lr_A.setRegion([self.processCurve.pointA, self.processCurve.pointA+2])

        

        #self.line1 = self.p1.addItem(self.lr)
        #self.line1 = self.p1.addItem(self.lr_A)
        #print("type",type(self.line1))
        
        self.p2.clearPlots()
        self.line2 = self.p2.plot(self.xx[self.processCurve.pointB-1:self.processCurve.pointC], self.yy[self.processCurve.pointB-1:self.processCurve.pointC],pen=pg.mkPen('c',width=1))
        

        ####设置滑动条B
        self.SliderA.setValue(self.processCurve.pointA)
        
        ####设置滑动条B
        self.SliderB.setValue(self.processCurve.pointB)

        ####设置滑动条C
        self.SliderC.setValue(self.processCurve.pointC)

        
        self.textEditA.setText(str(self.processCurve.pointA))
        self.textEditB.setText(str(self.processCurve.pointB))
        self.textEditC.setText(str(self.processCurve.pointC))


        # 曲线修正获得bc段
        self.processCurve.get_df_ACor_bc(self.params)
        
        self.processCurve.make_smooth(500)
        self.SliderAmp.setValue(500)
        self.textEditError.setText(str(round(self.processCurve.smoothErrorValue*100,4)))
        # print(self.processCurve.smoothErrorValue)

        #self.p2.clear()
        # 画平滑后的BC段曲线
        self.smoothBCLine = self.p2.plot(self.xx[self.processCurve.pointB-1:self.processCurve.pointC],self.processCurve.dfSmooth,pen=pg.mkPen('b',width=2))

        # 计算公式
        self.processCurve.compute_formual()

        # 绘制计算结果图
        self.draw_result()

       

    def init_disable(self, check, ckeck_up, ckeck_down):
        #上半操作区域
        if(ckeck_up):
            self.ButtonReset.setEnabled(check)
            # self.actionSave.setEnabled(check)
            # self.actionExport.setEnabled(check)
            #self.Messageshow.setEnabled(check)
            self.textEditProcessLeft.setEnabled(check)
            self.textEditProcessRight.setEnabled(check)
            self.Button_Change_Region.setEnabled(check)
            
            self.ButtonFitCurve.setEnabled(check)
            self.textEdit_FitCurve.setEnabled(check)
            # self.textEditParameter2.setEnabled(check)
            # self.textEditParameter3.setEnabled(check)
            
            self.ButtonMake.setEnabled(check)

            #p1,p2,p3三个按钮
            # self.ButtonChangeParameter1.setEnabled(check)
            # self.ButtonChangeParameter2.setEnabled(check)
            # self.ButtonChangeParameter3.setEnabled(check)

            self.textEditA.setEnabled(check)
            self.textEditB.setEnabled(check)
            self.textEditC.setEnabled(check)

            self.SliderA.setEnabled(check)
            self.SliderB.setEnabled(check)
            self.SliderC.setEnabled(check)
            

            self.ButtonBLeft.setEnabled(check)
            self.ButtonCLeft.setEnabled(check)
            self.ButtonBRight.setEnabled(check)
            self.ButtonCRight.setEnabled(check)
            self.ButtonA.setEnabled(check)
            self.ButtonB.setEnabled(check)
            self.ButtonC.setEnabled(check)
            
        #下半操作区域
        # 在曲线处理区域选择完后再能点击
        if(ckeck_down):
            
            self.SliderAmp.setEnabled(check)
            self.sample_start_Slider.setEnabled(check)
            self.sample_end_Slider.setEnabled(check)
            
            self.comboBox_4.setEnabled(check)
            self.lineEdit_4.setEnabled(check)
            self.lineEdit_5.setEnabled(check)
            self.lineEdit_6.setEnabled(check)
            self.lineEdit_7.setEnabled(check)
            self.lineEdit_8.setEnabled(check)
            self.textEditParameter1.setEnabled(check)

            
            self.parametersMakeButton.setEnabled(check)
            

    # 改变数据处理范围 即BC段处理范围
    def click_Button_Change_Region(self):
        self.Messageshow.setText("改变处理范围")
        self.params = [int(self.textEditProcessLeft.text()), int(self.textEditProcessRight.text())]
        self.isChangedPoint = True


    def click_ButtonBLeft(self):
        self.Messageshow.setText("B点左移10")
        self.init_disable(False, 0, 1)
        
               
        size = self.SliderB.value() - 10
        self.SliderB.setValue(size)
        self.lr.setRegion([self.SliderB.value(), self.SliderC.value()])


    def click_ButtonBRight(self):
        self.Messageshow.setText("B点右移10")
        self.init_disable(False, 0, 1)
               
        size = self.SliderB.value() + 10
        self.SliderB.setValue(size)
        self.lr.setRegion([self.SliderB.value(), self.SliderC.value()])
    

    def click_ButtonCLeft(self):
        self.Messageshow.setText("C点左移10")
        self.init_disable(False, 0, 1)
               
        size = self.SliderC.value() - 10
        self.SliderC.setValue(size)
        self.lr.setRegion([self.SliderB.value(), self.SliderC.value()])
        

    def click_ButtonCRight(self):
        self.Messageshow.setText("C点右移10")
        self.init_disable(False, 0, 1)
               
        size = self.SliderC.value() + 10
        self.SliderC.setValue(size)
        self.lr.setRegion([self.SliderB.value(), self.SliderC.value()])
    

    def click_ButtonA(self):
        self.Messageshow.setText("改变A点")
        self.init_disable(False, 0, 1)
               
        size = self.textEditA.text()
        self.SliderA.setValue(int(size))
        print("c",int(self.textEditA.text()))
        self.lr_A.setRegion([self.SliderA.value(), self.SliderA.value()])
        self.isChangedPoint = True

    def click_ButtonB(self):
        self.Messageshow.setText("改变B点")
        self.init_disable(False, 0, 1)
               
        size = self.textEditB.text()
        self.SliderB.setValue(int(size))
        print("c",int(self.textEditB.text()))
        self.lr.setRegion([self.SliderB.value(), self.SliderC.value()])
        self.isChangedPoint = True

    def click_ButtonC(self):
        self.Messageshow.setText("改变C点")
        self.init_disable(False, 0, 1)
               
        size = self.textEditC.text()
        self.SliderC.setValue(int(size))
        print("c",int(self.textEditC.text()))
        self.lr.setRegion([self.SliderB.value(), self.SliderC.value()])
        # 点改变，重新获取df,A,B,C
        self.isChangedPoint = True

    

    # 移动图中直线
    def changedHistogramSelection(self):
        #self.Messageshow.setText("改变区域选择区域")
        self.init_disable(False, 0, 1)
               
        value = self.lr.getRegion()
        tmp_B = int(value[0])
        tmp_C = int(value[1])
        
        self.textEditB.setText(str(tmp_B))
        self.textEditC.setText(str(tmp_C))
        #self.SliderB.setValue(tmp_B)
        #self.SliderC.setValue(tmp_C)
        # print(tmp_B, tmp_C)
    
    def changedfitsample(self):
        value = self.lr_fit_sample.getRegion()
        fit_sample_start = int(value[0])
        fit_sample_end = int(value[1])
        # self.lineEdit_fit_sample_start
        self.lineEdit_sample_start.setText(str(fit_sample_start))
        self.lineEdit_sample_end.setText(str(fit_sample_end))
    

    
    def draw_matplot(self, save_path):
        # 创建画布和主图 matplotlib

        plt.rcParams['font.family']=['SimHei']
        plt.rcParams["axes.unicode_minus"]=False #正常显示负号

        fig = plt.figure(figsize=(12, 8))
        # ax_main = fig.add_subplot(1, 1, 1)

        ax_sub0 = fig.add_axes([0.12,0.79,0.28, 0.1])
        ax_sub0.text(0.35,1,self.processCurve.fileName)
        ax_sub0.axis('off')

        # 创建子图1，并指定位置和大小
        ax_sub1 = fig.add_axes([0.12,0.74,0.28, 0.1])
        
        # 创建子图2，并指定位置和大小
        ax_sub2 = fig.add_axes([0.12 ,0.55, 0.28, 0.1])

        # 创建子图3，并指定位置和大小
        ax_sub3 = fig.add_axes([0.12,0.36, 0.28,0.1])

        # 创建子图4，并指定位置和大小
        ax_sub4 = fig.add_axes([0.12, 0.15, 0.28, 0.12])

        # 创建子图5，并指定位置和大小
        ax_sub5 = fig.add_axes([0.54, 0.56, 0.28 , 0.3])

        # 环境参数，计算力结果
        ax_sub6 = fig.add_axes([0.82,0.64,0.1,0.2])

        # 实验结果表格
        ax_sub7 = fig.add_axes([0.54,0.12,0.4,0.3])


        N =self.processCurve.N

        y = self.processCurve.primitiveCurve
        y = y-np.mean(y[0:self.processCurve.zeroLinelength])
        dt = self.processCurve.dT*1000
        t = np.array([dt * i for i in range(len(y))])

        ymin = min(y)
        i1= np.argmin(y)
        ymax = max(y)
        i2= np.argmax(y)
        if abs(ymax) < abs(ymin):
            ym = ymin
            i = i1
        else:
            ym = ymax
            i = i2
        yran = np.array([-abs(ym * 1.2), abs(ym * 1.2)])
        x = N * dt
        xran = np.append(np.array([0, x]),yran) #把yran的两个元素拼接在0和x之后，即xran数组中有4个元素
        
        ax_sub1.axis(xran)
        ax_sub1.set_xticks(np.linspace(0, x, 6, endpoint=True))

        ax_sub1.plot(t,y, linewidth= 1)
        ax_sub1.axhline(y=0,linewidth= 1)
        ax_sub1.set_ylabel('加速度(g)')
        ax_sub1.set_xlabel('时间(ms)')

        # 画首次碰撞加速度曲线
        a = self.processCurve.ACor1/self.processCurve.g
        t = np.array([dt * i for i in range(len(a))])
        t = t+20

        ymin = min(a)
        i1 = np.argmin(a)
        ymax = max(a)
        i2 = np.argmax(a)
        if abs(ymax) < abs(ymin):
            ym = ymin
            i = i1
        else:
            ym = ymax
            i = i2

        yran = np.array([-abs(ym * 1.2), abs(ym * 1.2)])
        
        N = np.round(N / 3)
        x = np.round(N * dt)
        xran = np.append(np.array([0, x]),yran)

        ax_sub2.axis(xran)
        ax_sub2.set_xticks(np.linspace(0, x, 6, endpoint=True))


        ax_sub2.plot(t,a,linewidth= 1)
        ax_sub2.axhline(y=0,linewidth= 1)
        ax_sub2.set_ylabel('加速度(g)')
        ax_sub2.set_xlabel('时间(ms)')

        # 首次碰撞速度曲线
        v = self.processCurve.V
        t = np.array([dt*i for i in range(len(v))])
        t = t+20 

        vm = max(v)
        i = np.argmax(v)
        yran = np.array([-abs(vm) * 1.2, abs(vm) * 1.2])
        xran = np.append(np.array([0, x]),yran)
        

        ax_sub3.axis(xran)
        ax_sub3.set_xticks(np.linspace(0, x, 6, endpoint=True))

        ax_sub3.plot(t,v,linewidth= 1) 
        ax_sub3.axhline(y=0,linewidth= 1)
        ax_sub3.set_ylabel('速度(m/s)')
        ax_sub3.set_xlabel('时间(ms)')


        # 首次碰撞位移曲线
        d = self.processCurve.D*1000
        t = np.array([dt*i  for i in range(len(d))])
        t = t+20

        smin = min(d)
        i1 = np.argmin(d)
        smax = max(d)
        i2 = np.argmax(d)
        if abs(smax) < abs(smin):
            sm = smin
            i = i1
        else:
            sm = smax
            i = i2
        yran = np.array([-abs(sm) * 1.2, abs(sm) * 1.2])
        xran = np.append(np.array([0, x]),yran)

        ax_sub4.axis(xran)
        ax_sub4.set_xticks(np.linspace(0, x, 6, endpoint=True))

        ax_sub4.plot(t,d,linewidth= 1)
        ax_sub4.axhline(y=0,linewidth= 1)
        ax_sub4.set_ylabel('位移(mm)')
        ax_sub4.set_xlabel('时间(ms)')

        # 冲击力曲线
        f = self.processCurve.F
        Nd = len(d)
        Nf = len(f)
        if Nd < Nf:
            d = d[0:Nd]
            f = f[0:Nd]
        else:
            d = d[0:Nf]
            f = f[0:Nf]

        # print("d.shape:",d.shape)
        # print("f.shape",f.shape)
        ax_sub5.plot(d,f,linewidth= 1)
        ax_sub5.set_ylabel('冲击力(kN)')
        ax_sub5.set_xlabel('冲击位移(mm)')

        # 型号处理分析曲线
        F1 = self.processCurve.F1

        ax_sub5.plot(F1[:,0],F1[:,1],marker='*',linewidth= 2)

        # 参数结果
        text_list=['环境温度:'+str(self.processCurve.At)+'℃\n',
                   '落锤高度:'+str(self.processCurve.H)+' cm\n',
                   '锤头质量:'+str(self.processCurve.M)+' kg\n',
                   'Am='+str(self.processCurve.Am)+' g\n',
                   'Vo='+str(self.processCurve.Vo)+' m/s\n',
                   'Dm='+str(self.processCurve.Dm)+' mm\n',
                   'Fm='+str(self.processCurve.Fm)+' kN\n',
                   'Wm='+str(self.processCurve.Wm)+' J\n',
                   'Wa='+str(self.processCurve.Wa)+' J\n',
                   'Kp='+str(self.processCurve.Kp)+' kN/m\n',
                   'Ke='+str(self.processCurve.Ke)+' kN/m\n',
                   'C='+str(self.processCurve.C)+'N.s/m\n',
                   'SF='+str(self.processCurve.SF)+'\n',
                   'n='+str(self.processCurve.n)+'\n']

        
        text_all =''
        for text in text_list:
            text_all += text

        ax_sub6.text(0.3,1,s=text_all,fontsize=10,horizontalalignment='left',verticalalignment='top')

        ax_sub6.axis('off')
        # 10等分表格
        
        col_labels = ['位移(mm)','抗力(kN)','分段刚度(kN/m)','上宽(kN)','下宽(kN)']
        # cellText =[]
        # len_row = len(self.processCurve.F1)
        # len_column = len(self.processCurve.F1[0])
        
        # for row in range(len_row):
        #     for column in range(len_column):
                
        #         cellText[row][column] = self.processCurve.F1[row][column]
        #   
        #         # print("执行了1")

        table = ax_sub7.table(cellText=self.processCurve.F1,
                      colLabels=col_labels, loc='center')
        # 设置边框颜色
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_edgecolor('#0000FF')
        
        ax_sub7.set_title("10等分位移——抗力对应值",loc='center')

        ax_sub7.axis('off')

        plt.savefig(save_path+ "/result.jpg",dpi=200)
        plt.show()



    def draw_result(self):
        """绘制计算结果图"""

        # 绘制p1-p7曲线]
        if self.isPlotResult==False:

            # self.draw_matplot()

           

            print("是否第一次画图",self.isPlotResult)
            # 画原始加速度曲线
            y = self.processCurve.primitiveCurve
            y = y-np.mean(y[0:self.processCurve.zeroLinelength])
            dt = self.processCurve.dT*1000
            t = np.array([dt * i for i in range(len(y))])
            
            self.primitiveAVelocityLine3 = self.p3.plot(t,y,pen=pg.mkPen('r',width=2))
            self.p3.setLabel('left',"加速度(g)")
            self.p3.setLabel('bottom','时间(ms)')



            # matplotlib
            

            # 画首次碰撞加速度曲线
            a = self.processCurve.ACor1/self.processCurve.g
            t = np.array([dt * i for i in range(len(a))])
            t = t+20

            self.firstCrashAVelocityLine4  =self.p4.plot(t,a,pen=pg.mkPen('r',width=2))
            self.p4.setLabel('left',"加速度(g)")
            self.p4.setLabel('bottom','时间(ms)')

        

            # 首次碰撞速度曲线
            v = self.processCurve.V
            t = np.array([dt*i for i in range(len(v))])
            t = t+20 

            self.firstCrashVelocityLine5 = self.p5.plot(t,v,pen=pg.mkPen('r',width=2))
            self.p5.setLabel('left',"速度(m/s)") 
            self.p5.setLabel('bottom','时间(ms)')
      
            # 首次碰撞位移曲线
            d = self.processCurve.D*1000
            t = np.array([dt*i  for i in range(len(d))])
            t = t+20

            self.firstCrashDisplaceLine6 = self.p6.plot(t,d,pen=pg.mkPen('r',width=2))
            self.p6.setLabel('left',"位移(mm)")
            self.p6.setLabel('bottom','时间(ms)')

       
            # 冲击力曲线
            f = self.processCurve.F
            Nd = len(d)
            Nf = len(f)
            if Nd < Nf:
                d = d[0:Nd]
                f = f[0:Nd]
            else:
                d = d[0:Nf]
                f = f[0:Nf]

            self.firstImpactForceLine7 = self.p7.plot(d,f,pen=pg.mkPen('r',width=2))
            # print("d.shape:",d.shape)
            # print("f.shape",f.shape)

            # 型号处理分析曲线
            F1 = self.processCurve.F1
            self.sharkDynamicChcteristic=self.p7.plot(F1[:,0],F1[:,1],pen=pg.mkPen('b',width=4),)
            self.sharkDynamicChcteristic.setSymbol("x")
            self.p7.setLabel('left',"冲击力(kN)")
            self.p7.setLabel('bottom','冲击位移(mm)')

         
            # 已经做了第一次计算处理，之后画图使用p.setData(),不使用p.plot()
            # 对全局变量进行更改
            self.isPlotResult=True

            
        
        else:

            # 原始加速度曲线
            y = self.processCurve.primitiveCurve
            y = y-np.mean(y[0:self.processCurve.zeroLinelength])
            dt = self.processCurve.dT*1000
            t = np.array([dt * i for i in range(len(y))])
            self.primitiveAVelocityLine3.setData(y)


            # 画首次碰撞加速度曲线
            a = self.processCurve.ACor1/self.processCurve.g
            t = np.array([dt * i for i in range(len(a))])
            t = t+20

            # self.firstCrashAVelocityLine  =self.p4.plot(t,a)
            self.firstCrashAVelocityLine4.setData(a)    
            # 首次碰撞速度曲线
            v = self.processCurve.V
            t = np.array([dt*i for i in range(len(v))])
            t = t+20 
            # self.firstCrashVelocityLine = self.p5.plot(t,v)
            self.firstCrashVelocityLine5.setData(v)

            # 首次碰撞位移曲线
            d = self.processCurve.D*1000
            t = np.array([dt*i  for i in range(len(d))])
            t = t+20

            # self.firstCrashDisplaceLine = self.p6.plot(t,d)
            self.firstCrashDisplaceLine6.setData(d)

            # 冲击力曲线
            f = self.processCurve.F
            Nd = len(d)
            Nf = len(f)
            if Nd < Nf:
                d = d[0:Nd]
                f = f[0:Nd]
            else:
                d = d[0:Nf]
                f = f[0:Nf]

            # self.FLine = self.p7.plot(d,f)
            self.firstImpactForceLine7.setData(d,f)
            # 信号处理分析曲线：
            F1 = self.processCurve.F1
            self.sharkDynamicChcteristic.setData(F1[:,0],F1[:,1])

         #给表格填充数据
        len_row = len(self.processCurve.F1)
        len_column = len(self.processCurve.F1[0])
        print("表格长度", len_row,len_column)
        for row in range(len_row):
            for column in range(len_column):
                
                item = QStandardItem(str(self.processCurve.F1[row][column]))
                item.setTextAlignment(Qt.AlignCenter)
                #给表格对应位置赋值
                self.tableModel.setItem(row,column,item)
                # print("执行了1")
        
        # 计算结果参数
        self.textEditP1.setText(str(self.processCurve.At))
        self.textEditP2.setText(str(self.processCurve.H))
        self.textEditP3.setText(str(self.processCurve.M))
        self.textEditP4.setText(str(self.processCurve.Am))
        self.textEditP5.setText(str(self.processCurve.Vo))
        self.textEditP6.setText(str(self.processCurve.Dm))
        self.textEditP7.setText(str(self.processCurve.Fm))
        self.textEditP8.setText(str(self.processCurve.Wm))
        self.textEditP9.setText(str(self.processCurve.Wa))
        self.textEditP10.setText(str(self.processCurve.Kp))
        self.textEditP11.setText(str(self.processCurve.Ke))
        self.textEditP12.setText(str(self.processCurve.C))
        self.textEditP13.setText(str(self.processCurve.SF))
        self.textEditP14.setText(str(self.processCurve.n))

    

    def click_parametersMakeButton(self):
        x = eval(self.textEditParameter1.text())
        g=eval(self.lineEdit_4.text())
        M=eval(self.lineEdit_5.text())
        AT = eval(self.lineEdit_6.text())
        H = eval(self.lineEdit_7.text())
        deviceN = eval(self.lineEdit_8.text())
        isCorrect = self.comboBox_4.currentIndex()
        print("isCoorect:",isCorrect)
        self.processCurve.set_parameters(x,g,M,AT,H,deviceN,isCorrect)
        self.processCurve.compute_formual()
        self.draw_result()
        
        

                 



    def click_buttonMake(self):
        # if(int(self.textEditA.text()) < self.params[0]):
        #     self.Messageshow.setText("A点不能小于处理区域左值！！！")
        #     return
        # if(int(self.textEditB.text()) > int(self.textEditC.text())):
        #     self.Messageshow.setText("B点的值不能大于C点！！！！！！")
        #     return
        # print(int(self.textEditA.text()))
        if int(self.textEditA.text()) < self.params[0] or int(self.textEditA.text()) >self.params[1]:
            QMessageBox.information(self, "tip", "曲线A点不在数据处理区间内，请重新选择B点,保证A点在"+str(self.params[0])+"~"+str(self.params[1])+"!" , QMessageBox.Yes) #最后的Yes表示弹框的按钮显示为Yes，默认按钮显示为OK,不填QMessageBox.Yes即为默认
            return 
        
        if int(self.textEditB.text()) < self.params[0] or int(self.textEditB.text()) > self.params[1]:
            QMessageBox.information(self, "tip", "曲线B点不在数据处理区间内，请重新选择B点，保证B点在"+str(self.params[0])+"~"+str(self.params[1])+"!", QMessageBox.Yes) #最后的Yes表示弹框的按钮显示为Yes，默认按钮显示为OK,不填QMessageBox.Yes即为默认
            return 
        
        if int(self.textEditC.text()) < self.params[0] or int(self.textEditC.text()) > self.params[1]:
            QMessageBox.information(self, "tip", "曲线C点不在数据处理区间内，请重新选择C点，保证C点在"+str(self.params[0])+"~"+str(self.params[1])+"!" , QMessageBox.Yes)
            return
        
        if int(self.textEditA.text()) > int(self.textEditB.text()) or int(self.textEditA.text()) > int(self.textEditC.text()):
            QMessageBox.information(self, "tip", "请保证A点位置小于B点和C点位置，请重新选择A点！" , QMessageBox.Yes)
            return
        
        if int(self.textEditB.text()) > int(self.textEditC.text()) :
            QMessageBox.information(self, "tip", "请保证B点位置小于C点位置，请重新选择B点！" , QMessageBox.Yes)
            return
        

        # if self.processCurve.pointA < self.params[0] or self.processCurve.pointA >self.params[1]:
        #     QMessageBox.information(self, "tip", "曲线A点不再数据处理区间内，请重新选择B点,保证A点在"+self.params[0]+"~"+self.params[1]+"!" , QMessageBox.Yes) #最后的Yes表示弹框的按钮显示为Yes，默认按钮显示为OK,不填QMessageBox.Yes即为默认
        #     return 
        
        # if self.processCurve.pointB < self.params[0] or self.processCurve.pointB > self.params[1]:
        #     QMessageBox.information(self, "tip", "曲线B点不再数据处理区间内，请重新选择B点，保证B点在"+self.params[0]+"~"+self.params[1]+"!", QMessageBox.Yes) #最后的Yes表示弹框的按钮显示为Yes，默认按钮显示为OK,不填QMessageBox.Yes即为默认
        #     return 
        
        # if self.processCurve.pointC < self.params[0] or self.processCurve.pointB > self.params[1]:
        #     QMessageBox.information(self, "tip", "曲线C点不再数据处理区间内，请重新选择C点，保证C点在"+self.params[0]+"~"+self.params[1]+"!" , QMessageBox.Yes)
        #     return
        
        # if self.processCurve.pointA > self.processCurve.pointB or self.processCurve.pointA > self.processCurve.pointC:
        #     QMessageBox.information(self, "tip", "请保证A点位置小于B点和C点位置，请重新选择A点！" , QMessageBox.Yes)
        #     return
        
        # if self.processCurve.pointB > self.processCurve.pointC :
        #     QMessageBox.information(self, "tip", "请保证B点位置小于C点位置，请重新选择B点！" , QMessageBox.Yes)
        #     return
        


        

        # print("a,b,c",int(self.textEditA.text()),int(self.textEditB.text()),int(self.textEditC.text()))
        self.processCurve.set_pointABC(int(self.textEditA.text()),int(self.textEditB.text()),int(self.textEditC.text()))
        print("a,b,c",self.processCurve.pointA,self.processCurve.pointB,self.processCurve.pointC)
        self.processCurve.get_df_ACor_bc(self.params)

        self.line2.setData(self.xx[self.processCurve.pointB-1:self.processCurve.pointC], self.yy[self.processCurve.pointB-1:self.processCurve.pointC])

        self.processCurve.make_smooth(500)
        # self.processCurve.get_sugget_smooth()
        # self.sliderAmp
        #更新平滑的曲线
        self.smoothBCLine.setData(self.xx[self.processCurve.pointB-1:self.processCurve.pointC],self.processCurve.dfSmooth)
    
        # self.processCurve.compute_formual()
        # 计算公式
        # self.processCurve.compute_formual()

        # self.draw_result()

        self.init_disable(True, 0, 1)
        self.Messageshow.setText("Make Done！！！！")
    

    # 选择文件按钮
    def click_fileSelectButton(self):
        filePath = QFileDialog.getExistingDirectory(self,"文件保存",'.')
        
        filelist = filePath.split('/')
        # os.sep：文件路径斜杠
        filePath=filelist[0]+os.sep
        for i in range(1,len(filelist)):
            filePath=os.path.join(filePath,filelist[i])
        print(filePath)
        self.lineEdit_3.setText(filePath)
        

    ### 保存选项确定点击按钮###
    def click_saveOptionButton(self):
        experimentType=str(self.lineEdit.text())
        experimentDirect=self.comboBox.currentText()
        experimentLoadCoefficient=self.comboBox_2.currentText()
        expreimentPrePressing=self.comboBox_3.currentText()
        baseSavePath = self.lineEdit_3.text()
        print("基础路径",baseSavePath)

        # self.draw_matplot(baseSavePath)
        

        # baseSavePath=os.path.join(os.path.dirname(baseSavePath),os.path.basename(baseSavePath))
        savePath=os.path.join(baseSavePath,experimentType)
        if not os.path.exists(savePath):
            print('不存在路径，创建文件')
            os.mkdir(savePath)

        savePath=os.path.join(savePath,experimentDirect)
        if not os.path.exists(savePath):
            print('不存在路径，创建文件')
            os.mkdir(savePath)
        savePath=os.path.join(savePath,experimentLoadCoefficient)
        if not os.path.exists(savePath):
            print('不存在路径，创建文件')
            os.mkdir(savePath)
        savePath=os.path.join(savePath,expreimentPrePressing)
        if not os.path.exists(savePath):
            print('不存在路径，创建文件')
            os.mkdir(savePath)
        savePath=os.path.join(savePath,self.processCurve.fileName)
        if not os.path.exists(savePath):
            print('不存在路径，创建文件')
            os.mkdir(savePath)
        self.lineEdit_2.setText(savePath)
        print("最终路径:",savePath)
    
     ### 保存选项确定点击按钮###
    def click_saveMakeButton(self):

      # 文件夹不存在，则创建文件夹
        savePath = self.lineEdit_2.text()
        print("当前时间",str(time.time()))
        # curTime =datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        # savePath=os.path.join(savePath,curTime)
        if not os.path.exists(savePath):
            print('不存在路径，创建文件')
            os.mkdir(savePath)
        

        # 实验后的曲线用matplot 画图并保存
        self.draw_matplot(savePath)
        pixmap = self.groupBox_5.grab()
        # current_widget = QWidget()
        # current_widget.setLayout(self.groupBox_4)
        # current_widget.show()
        # pixmap = current_widget.grab()
        # pixmap = QPixmap.grabWidget(self.verticalLayout_6)
        pixmap.save(savePath+'/'+'all_result.jpg')

        pixmap_all = self.Form.grab()
        pixmap_all.save(savePath+'/'+'all_page.jpg')

        
        # 保存原始加速度曲线
        t,y = self.primitiveAVelocityLine3.getData()
        primitiveAvelocityLine = np.c_[t,y]
        pd.DataFrame(primitiveAvelocityLine).to_csv(savePath+'/'+'primitiveAcceleration.csv',header=None,index=None)
        pd.DataFrame(primitiveAvelocityLine).to_csv(savePath+'/'+'primitiveAcceleration.txt',header=None,index=None)
        primitiveAvelocityLineExporter = pyqtgraph.exporters.ImageExporter(self.p3)
        primitiveAvelocityLineExporter.params.param('width').setValue(1096)
        primitiveAvelocityLineExporter.params.param('height').setValue(596)
        primitiveAvelocityLineExporter.export(savePath+'/'+'primitiveAcceleration.jpg')
        primitiveAvelocityLineExporter.export(savePath+'/'+'primitiveAcceleration.bmp')
        # ex = pyqtgraph.exporters.ImageExporter(w.scene())
        # ex.export(fileName="test.png")

        # 保存冲击加速度曲线
        t,y = self.firstCrashAVelocityLine4.getData()
        firstCrashAVelocityLine = np.c_[t,y]
        pd.DataFrame(firstCrashAVelocityLine).to_csv(savePath+'/'+'sharkAcceleration.csv',header=None,index=None)
        pd.DataFrame(firstCrashAVelocityLine).to_csv(savePath+'/'+'sharkAcceleration.txt',header=None,index=None)
        firstCrashAVelocityLineExporter = pyqtgraph.exporters.ImageExporter(self.p4)
        firstCrashAVelocityLineExporter.params.param('width').setValue(1096)
        firstCrashAVelocityLineExporter.params.param('height').setValue(596)
        firstCrashAVelocityLineExporter.export(savePath+'/'+'sharkAcceleration.jpg')
        firstCrashAVelocityLineExporter.export(savePath+'/'+'sharkAcceleration.bmp')

        # 保存冲击速度曲线
        t,y = self.firstCrashVelocityLine5.getData()
        firstCrashVelocityLine = np.c_[t,y]
        pd.DataFrame(firstCrashVelocityLine).to_csv(savePath+'/'+'sharkVelocity.csv',header=None,index=None)
        pd.DataFrame(firstCrashVelocityLine).to_csv(savePath+'/'+'sharkVelocity.txt',header=None,index=None)
        firstCrashVelocityLineExporter = pyqtgraph.exporters.ImageExporter(self.p5)
        firstCrashVelocityLineExporter.params.param('width').setValue(1096)
        firstCrashVelocityLineExporter.params.param('height').setValue(596)
        firstCrashVelocityLineExporter.export(savePath+'/'+'sharkVelocity.jpg')
        firstCrashVelocityLineExporter.export(savePath+'/'+'sharkVelocity.bmp')

        # 保存冲击位移曲线
        t,y = self.firstCrashDisplaceLine6.getData()
        firstCrashDisplaceLine = np.c_[t,y]
        pd.DataFrame(firstCrashDisplaceLine).to_csv(savePath+'/'+'sharkDisplacement.csv',header=None,index=None)
        pd.DataFrame(firstCrashDisplaceLine).to_csv(savePath+'/'+'sharkDisplacement.csv',header=None,index=None)
        firstCrashDisplaceLineExporter = pyqtgraph.exporters.ImageExporter(self.p6)
        firstCrashDisplaceLineExporter.params.param('width').setValue(1096)
        firstCrashDisplaceLineExporter.params.param('height').setValue(596)
        firstCrashDisplaceLineExporter.export(savePath+'/'+'sharkDisplacement.jpg')
        firstCrashDisplaceLineExporter.export(savePath+'/'+'sharkDisplacement.bmp')
        

        # 保存冲击动态特性曲线
        t,y = self.firstImpactForceLine7.getData()
        firstImpactForceLine= np.c_[t,y]
        pd.DataFrame(firstImpactForceLine).to_csv(savePath+'/'+'sharkDynamicCharacteristic.csv',header=None,index=None)
        pd.DataFrame(firstImpactForceLine).to_csv(savePath+'/'+'sharkDynamicCharacteristic.txt',header=None,index=None)
        firstImpactForceLineExporter = pyqtgraph.exporters.ImageExporter(self.p7)
        firstImpactForceLineExporter.params.param('width').setValue(1680)
        firstImpactForceLineExporter.params.param('height').setValue(1284)
        firstImpactForceLineExporter.export(savePath+'/'+'sharkDynamicCharacteristic.jpg')
        firstImpactForceLineExporter.export(savePath+'/'+'sharkDynamicCharacteristic.bmp')

        

        # 保存 结果1 csv
        #print("表格")
        #print(self.processCurve.F1)
        # 冲击特性参数
       
        dataDict={
            '位移(mm)':self.processCurve.F1[:,0].reshape(-1),
            '抗力(kN)':self.processCurve.F1[:,1].reshape(-1),
            '分段刚度(kN/m)':self.processCurve.F1[:,2].reshape(-1),
            '上宽(kN)':self.processCurve.F1[:,3].reshape(-1),
            '下宽(kN)':self.processCurve.F1[:,4].reshape(-1)   
        }
        dataHeader=['位移(mm)','抗力(kN)','分段刚度(kN/m)','上宽(kN)','下宽(kN)']
        # print("dataDict:",dataDict)
        pd.DataFrame(dataDict).to_csv(savePath + '/' + 'sharkCharacteristicParameter.csv', columns=dataHeader,encoding='utf_8_sig', index=None)
        pd.DataFrame(dataDict).to_csv(savePath + '/' + 'sharkCharacteristicParameter.txt', columns=dataHeader,encoding='utf_8_sig', index=None)
        # pd.DataFrame(self.processCurve.F1).to_csv(savePath + '/' + 'sharkCharacteristicParameter.csv', columns=['位移(mm)','抗力(kN)','分段刚度(kN/m)','上宽(kN)','下宽(kN)'],encoding='utf_8_sig', index=None)
        
        # 保存 结果2 csv
        Data2 = np.array([self.processCurve.At,
                         self.processCurve.H,
                         self.processCurve.M,
                         self.processCurve.Am,
                         self.processCurve.Vo,
                         self.processCurve.Dm,
                         self.processCurve.Fm,
                         self.processCurve.Wm,
                         self.processCurve.Wa,
                         self.processCurve.Kp,
                         self.processCurve.Ke,
                         self.processCurve.C,
                         self.processCurve.SF,
                         self.processCurve.n])
        
        Data2index=["环境温度(℃)","落锤高度(cm)","锤头质量(kg)",
                    "Am(g)","Vo(m/s)","Dm(mm)","Fm(kN)","Wm(J)",
                    "Wa(J)","Kp(kN/m)","Ke(kN/m)","C(N·s/m)","SF","η"]
        #print(Data2)
        pd.DataFrame(Data2,index=Data2index).to_csv(savePath + '/' + 'EnvironmentalParameterAndExperimentalResult.csv',encoding="utf_8_sig", header=None)
        pd.DataFrame(Data2,index=Data2index).to_csv(savePath + '/' + 'EnvironmentalParameterAndExperimentalResult.txt',encoding="utf_8_sig", header=None)
        QMessageBox.information(self, "tip", "数据已保存到"+ savePath , QMessageBox.Yes) #最后的Yes表示弹框的按钮显示为Yes，默认按钮显示为OK,不填QMessageBox.Yes即为默认
  

    #### 滑条B值变化
    def SliderValueChangedSlot_B(self):
        self.Messageshow.setText("滑动B滑动条")
        self.init_disable(False, 0, 1)

        size = int(self.SliderB.value())
        self.lr.setRegion([self.SliderB.value()-1,self.SliderC.value()])
        self.textEditB.setText(str(size))
        print(self.lr.getRegion())

    #### 滑条A值变化
    def SliderValueChangedSlot_A(self): 
        self.Messageshow.setText("滑动A滑动条")
        self.init_disable(False, 0, 1)
        self.lr_A.setRegion([self.SliderA.value(),self.SliderA.value()+3])             
        size = int(self.SliderA.value())
        self.textEditA.setText(str(size))
        #print(self.lr.getRegion())

    #### 滑条C值变化
    def SliderValueChangedSlot_C(self): 
        self.Messageshow.setText("滑动C滑动条")
        self.init_disable(False, 0, 1)
                     
        size = self.SliderC.value()
        self.lr.setRegion([self.SliderB.value()-1,self.SliderC.value()])
        # self.point_b = size
        self.textEditC.setText(str(size))
        print(self.lr.getRegion())

    #### 滑动条Amp值变化
    def SliderValueChangedSlot_Amp(self):
        # 重新获得dfsmooth
        self.textEditSmoothIter.setText(str(self.SliderAmp.value()))
        self.processCurve.make_smooth(self.SliderAmp.value())
        self.textEditError.setText(str(round(self.processCurve.smoothErrorValue*100,4)))
        # self.textEditError.setText(self.)
        # self.processCurve.compute_formual()
        # A = self.processCurve.A[self.processCurve.startPoint-1:self.processCurve.endPoint]
        # A = np.array(A)
        # print(A)
        if(self.isSmooth):
            self.smoothBCLine.setData(self.xx[self.processCurve.pointB-1:self.processCurve.pointC],self.processCurve.dfSmooth)
        
        # 计算公式
        if self.isPlotResult:
            self.processCurve.compute_formual()
            #绘制计算结果图
            self.draw_result()

    def SliderValueChangedSlot_SampleStart(self):

        self.lr_fit_sample.setRegion([self.sample_start_Slider.value(),self.sample_end_Slider.value()])
        self.lineEdit_sample_start.setText(str(self.sample_start_Slider.value()))
    
    def SliderValueChangedSlot_SampleEnd(self):
        self.lr_fit_sample.setRegion([self.sample_start_Slider.value(), self.sample_end_Slider.value()])
        self.lineEdit_sample_end.setText(str(self.sample_end_Slider.value()))



    def mouseMoved1(self,evt):
            data1 = self.processCurve.df['data'][self.processCurve.pointB-1:self.processCurve.pointC]
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if self.p2.sceneBoundingRect().contains(pos):
                mousePoint = self.p2.vb.mapSceneToView(pos)
                index = int(mousePoint.x())
                point = index
                if(index > data1.index[-1]):
                    point = data1.index[-1]        
                if index > 0 and index < len(data1) + self.processCurve.pointB:
                    self.label_cursor2.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>" % (mousePoint.x(), point))
                self.vLine2.setPos(mousePoint.x())
                self.hLine2.setPos(point)        

    def mouseMoved(self,evt):
            data1 = self.processCurve.df['data']
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if self.p1.p().contains(pos):
                mousePoint = self.p1.vb.mapSceneToView(pos)
                index = int(mousePoint.x())
                point = index
                if(index > data1.index[-1]):
                    point = data1.index[-1]
                if index > 0 and index < len(data1):
                    self.label_cursor.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>" % (mousePoint.x(), point))
                self.vLine.setPos(mousePoint.x())
                self.hLine.setPos(point)

    def set_graph_ui(self):

        pg.setConfigOptions(antialias=True)  # pg全局变量设置函数，antialias=True开启曲线抗锯齿

        win1 = pg.GraphicsLayoutWidget()  # 创建pg layout，可实现数据界面布局自动管理
        win2 = pg.GraphicsLayoutWidget()
        win3 = pg.GraphicsLayoutWidget()
        win4 = pg.GraphicsLayoutWidget()
        win5 = pg.GraphicsLayoutWidget()
        win6 = pg.GraphicsLayoutWidget()
        win7 = pg.GraphicsLayoutWidget()
    
        win1.setBackground('w')
        win2.setBackground('w')
        win3.setBackground('w')
        win4.setBackground('w')
        win5.setBackground('w')
        win6.setBackground('w')
        win7.setBackground('w')


        #鼠标交互文本
        self.label_cursor = pg.LabelItem(justify='right')
        win1.addItem(self.label_cursor)
        self.label_cursor2 = pg.LabelItem(justify='right')
        win2.addItem(self.label_cursor2)

        # pg绘图窗口可以作为一个widget添加到GUI中的graph_layout，当然也可以添加到Qt其他所有的容器中
        self.MainFigure1.addWidget(win1)
        self.MainFigure2.addWidget(win2)
        self.Figure1.addWidget(win3)
        self.Figure2.addWidget(win4)
        self.Figure3.addWidget(win5)
        self.Figure4.addWidget(win6)
        self.Figure5.addWidget(win7)
        
        
        
        p1 = win1.addPlot(row=1, col=0)  # 添加第一个绘图窗口
        p1.showGrid(x=True, y=True)  # 栅格设置函数
        #print(type(p1))
        # p1.setBackground('w')

        p2 = win2.addPlot(row=1, col=0,title='')  # 添加第一个绘图窗口
        
        p2.showGrid(x=True, y=True)  # 栅格设置函数

        p3 = win3.addPlot()  # 添加第一个绘图窗口
        p3.showGrid(x=True, y=True)  # 栅格设置函数

        p4 = win4.addPlot()  # 添加第一个绘图窗口
        p4.showGrid(x=True, y=True)  # 栅格设置函数

        p5 = win5.addPlot()  # 添加第一个绘图窗口
        p5.showGrid(x=True, y=True)  # 栅格设置函数

        p6 = win6.addPlot()  # 添加第一个绘图窗口
        p6.showGrid(x=True, y=True)  # 栅格设置函数

        p7 = win7.addPlot()  # 添加第一个绘图窗口
        p7.showGrid(x=True, y=True)  # 栅格设置函数

        #cross hair鼠标交互
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        p1.addItem(self.vLine, ignoreBounds=True)
        p1.addItem(self.hLine, ignoreBounds=True)
        #cross hair
        self.vLine2 = pg.InfiniteLine(angle=90, movable=False)
        self.hLine2 = pg.InfiniteLine(angle=0, movable=False)
        p2.addItem(self.vLine2, ignoreBounds=True)
        p2.addItem(self.hLine2, ignoreBounds=True)

        return p1, p2, p3, p4, p5, p6, p7
    
    ### 打开文件###
    def processInit(self,file):
        #打开执行的函数体
        # self.message_dict = message_dict
        self.flag1 = 1
        self.isPlotResult=False
        # self.file = self.message_dict['path']
        self.file = file
        self.init_disable(False, 1, 1)

        # self.file,ok=QFileDialog.getOpenFileName(self,"打开",".","Text Files(*.txt *.dat *.csv);;All Files (*)")

        self.p1.clearPlots()
        self.p2.clearPlots()
        self.p3.clearPlots()
        self.p4.clearPlots()
        self.p5.clearPlots()
        self.p6.clearPlots()
        self.p7.clearPlots()
        self.lineEdit_2.clear()
        # 创建self.processCurve对象
        # self.processCurve = self.processCurve()
        # 可以获得self.df
        self.processCurve.set_df(self.file)
        
        # 曲线拟合
        # self.processCurve.fit_tail()


        #确定A,B,C点
        self.processCurve.determine_a_b_c()



        self.xx = np.array(self.processCurve.df['data'].index)
        self.yy = np.array(self.processCurve.df['data'])
        df_max = max(self.processCurve.df['data'])
        df_min = min(self.processCurve.df['data'])

        #绘图
        self.line0=self.p1.plot(self.xx, self.yy,pen=pg.mkPen('b',width=2))
        # 绘制拟合曲线
        # startP=8400
        # endP=9400
        

        # fitData=np.array(self.processCurve.fitdf.data)
        # self.line1= self.p1.plot(self.xx,fitData,pen=pg.mkPen('r',width=2))
        # self.line1= self.p1.plot(self.xx,self.processCurve.fitData,pen=pg.mkPen('r',width=3))
        # 添加一个线区域选择项目，起始区间在B到C
        # self.lr = pg.LinearRegionItem([self.processCurve.pointB-1, self.processCurve.pointC])
        # self.lr_A = pg.LinearRegionItem([self.processCurve.pointA, self.processCurve.pointA+2], movable=False)
        self.lr.setRegion([self.processCurve.pointB-1,self.processCurve.pointC])
        self.lr_A.setRegion([self.processCurve.pointA,self.processCurve.pointA+2])
        #rangeslider滑动
        self.lr.sigRegionChanged.connect(self.changedHistogramSelection)
        #self.lr.sigRegionChangeFinished.connect(self.changedHistogramSelection)

        self.linelr = self.p1.addItem(self.lr)
        self.linelr_A = self.p1.addItem(self.lr_A)
        #print("type",type(self.line1))

        self.line2 = self.p2.plot(self.xx[self.processCurve.pointB-1:self.processCurve.pointC], self.yy[self.processCurve.pointB-1:self.processCurve.pointC],pen=pg.mkPen('b',width=1))
        
        



        ####设置滑动条A
        #最小值
        self.SliderA.setMinimum(0)
        #设置最大值
        self.SliderA.setMaximum(len(self.processCurve.df))
        #步长
        self.SliderA.setSingleStep(10)
        self.SliderA.setValue(self.processCurve.pointA)
        
        ####设置滑动条B
        #最小值
        self.SliderB.setMinimum(0)
        #设置最大值
        self.SliderB.setMaximum(len(self.processCurve.df))
        #步长
        self.SliderB.setSingleStep(10)
        self.SliderB.setValue(self.processCurve.pointB)

        ####设置滑动条C
        #最小值
        self.SliderC.setMinimum(0)
        #设置最大值
        self.SliderC.setMaximum(len(self.processCurve.df))
        #步长
        self.SliderC.setSingleStep(10)
        self.SliderC.setValue(self.processCurve.pointC)

        ####设置滑动条Amp
        # 最小值
        self.SliderAmp.setMinimum(0)
        self.SliderAmp.setMaximum(1000)
        # self.SliderC.setSingleStep(10)
        # self.SliderC.setTickInterval(5)

        # 设置拟合采样段滑动条
        #sample_start
        self.sample_start_Slider.setMinimum(self.processCurve.pointB)
        self.sample_start_Slider.setMaximum(self.processCurve.pointC)
        # sample_end
        self.sample_end_Slider.setMinimum(self.processCurve.pointB)
        self.sample_end_Slider.setMaximum(self.processCurve.pointC)

        # 拟合采样段
        sample_point =[7000,8000]
        self.sample_start_Slider.setValue(sample_point[0])
        self.sample_end_Slider.setValue(sample_point[1])
        self.lr_fit_sample.setRegion(sample_point)
        self.line_fit_sample = self.p2.addItem(self.lr_fit_sample)
        self.lr_fit_sample.sigRegionChanged.connect(self.changedfitsample)
        self.lineEdit_sample_start.setText(str(sample_point[0]))
        self.lineEdit_sample_end.setText(str(sample_point[1]))

        
        self.textEditA.setText(str(self.processCurve.pointA))
        self.textEditB.setText(str(self.processCurve.pointB))
        self.textEditC.setText(str(self.processCurve.pointC))


        # 曲线修正获得bc段
        self.processCurve.get_df_ACor_bc(self.params)
        self.processCurve.make_smooth(500)
        self.SliderAmp.setValue(500)
        self.textEditError.setText(str(round(self.processCurve.smoothErrorValue*100,4)))
        # print(self.processCurve.smoothErrorValue)

        
        

        
        # 画平滑后的BC段曲线
        #self.p2.clear()
        # self.xxLine = self.p2.plot(self.xx[self.processCurve.pointB-1:self.processCurve.pointC],pen=pg.mkPen('b',width=2))
        self.smoothBCLine = self.p2.plot(self.xx[self.processCurve.pointB-1:self.processCurve.pointC],self.processCurve.dfSmooth,pen=pg.mkPen('r',width=2))
        self.isSmooth=True
        # self.xxLine = self.p2.plot(self.xx[self.processCurve.pointB-1:self.processCurve.pointC],pen=pg.mkPen('b',width=2))
        # self.xx[self.processCurve.pointB-1:self.processCurve.pointC]

        # 计算公式
        # self.processCurve.compute_formual()

        # self.draw_result()

        self.init_disable(True, 1, 0)
        
        #鼠标cursor指针交互
        proxy = pg.SignalProxy(self.p1.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        proxy1 = pg.SignalProxy(self.p2.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved1)
        #print(self.params[0], self.params[1])
        #处理左右区间初始化
        self.textEditProcessLeft.setText(str(self.params[0]))
        self.textEditProcessRight.setText(str(self.params[1]))
        self.textEditParameter1.setText(str(1))
