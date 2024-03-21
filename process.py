import os
from scipy.signal import lfilter
# from scipy import signal
from scipy.signal import butter 
# from scipy.signal import savgol_filter
from scipy.io import loadmat
import numpy as np
# from numpy import cumsum, concatenate,

import pandas as pd

import matplotlib.pyplot as plt

# import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# import torch

from torch import from_numpy 

from deeplearn import predict

# import matplotlib.pyplot as plt


#---------图像平滑函数----------
def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

# --------对曲线各点进行求差值，得到一条差值曲线---------
def difference_value(dfIndex,y,gap):
    difference = []
    differneceIndex = []
    for i in range(dfIndex[0],dfIndex[-1]-gap):
        difference.append( (y[i]-y[i+gap])**2)
        differneceIndex.append(i)
    return difference,differneceIndex



class ProcessCurve:
    """
    1.set_df()
    2.determin_a_b_c()
    3.get_df_ACor_bc()
    4.make_smooth()
    5.compute_formual()
    6.get_result()

    """
    # def __init__(self):
    #     self.filePath = filePath
    #     self.df = pd.read_csv(self.filePath)
    #     self.A = self.df.copy()

    def set_df2(self,filePath,fileName):
        self.fileName = fileName
        df =pd.read_csv(filePath,header=None)
        df['time'] = df[0].str.split()
        df['tick'] = df['time'].map(lambda x:eval(x[0]))
        df['data'] = df['time'].map(lambda x:eval(x[1]))
        df.pop('time')
        self.df = df

    
    # 获取曲线数据，可以打开mat和dat格式
    def set_df(self,filePath):

        print("曲线路径：",filePath)
        # 获得文件名
        # 文件路径为反斜杠
        fileNameAndType = filePath.split('/')[-1]
        # 文件路径为正斜杠
        fileNameAndType = os.path.basename(filePath)
        self.fileNameAndType = fileNameAndType
        print(self.fileNameAndType)
        # 获取文件名(不带后缀)，文件类型
        fileName = fileNameAndType.split('.')[0]
        self.fileName = fileName
        fileType = fileNameAndType.split('.')[1]
        self.fileType = fileType
        if fileType=='mat':
            data=loadmat(filePath)
            df=data['Datas']
            # print(df)
            # print(df.shape)
            dataDict = {'data':df[:,0]}
            self.df=pd.DataFrame(dataDict)
            # print("Init df",self.df)
            # print("",data['SampleFrequency'])
            self.dT= 1 / eval(data['SampleFrequency'][0])
        
        elif fileType=='dat':
            df =pd.read_csv(filePath,header=None)
            print(df)
            df['time'] = df[0].str.split()
            df['tick'] = df['time'].map(lambda x:eval(x[0]))
            df['data'] = df['time'].map(lambda x:eval(x[1]))
            print(df)
            df.pop('time')
            self.df = df
            # 采样频率
            self.dT=(self.df['tick'][1] - self.df['tick'][0]) / 1000
            # self.df=df['data']
            # print("Init df\n",self.df)
        elif fileType=='txt':
            df =pd.read_csv(filePath,header=None)
            df['time'] = df[0].str.split()
            df['tick'] = df['time'].map(lambda x:eval(x[0]))
            df['data'] = df['time'].map(lambda x:eval(x[1]))
            df.pop('time')
            self.df = df
            # 采样频率
            self.dT=(self.df['tick'][1] - self.df['tick'][0]) / 1000
            # self.df=df['data']
            # print("Init df\n",self.df)


    def model_predict_a_b_c(self):
        
        # 转换为模型预测的数据格式
        x = np.arange(0,20000,1).reshape((1,1,20000))
        # x = x.reshape((20000,1))
        y = self.df['data'].to_numpy().reshape((1,1,20000))
        # 1*2*20000
        input = np.concatenate((x,y),axis=1) 

        # input 转换为tensor
        input = from_numpy(input).float()
        # input = torch.from_numpy(input).float()
        # 已训练好的模型
        # path = os.path.join("./","model_val_loss_180.pth")
        
        # model_path = "model_val_loss_180.pth"
        root = os.path.dirname(os.path.realpath(__file__))
        path =os.path.join(root,"model_val_loss_180.pth")
        pointA,pointB,pointC = predict(path,input)
        self.pointA= int(pointA)
        self.pointB= int(pointB)
        self.pointC = int(pointC)
        # plt.figure(figsize=(24,18))
        # plt.plot(self.df['data'],'g')
        # plt.axvline(self.pointA,c='r')
        # plt.axvline(self.pointB,c='r')
        # plt.axvline(self.pointC,c='r')
        # plt.savefig(self.fileName+'.jpg')
        # plt.show()


    
    def determine_a_b_c(self):
        """determine a,b,c"""
        ##############
        print("name:",self.fileName)
        # 方法2：
        df0 = self.df.copy()
        # print(df0)
        noisePoint= 100
        df0 = df0[100:] # 去掉初始噪声
        # print("df0",df0)
        dfData = df0['data']
        # print(df0)
        # a = dfData[1800:2800].argmin()
        boundaryAList= df0[abs(df0['data'])<0.1].index.to_list()
        for i in range(1,len(boundaryAList)):
            # plt.axvline(boundaryAList[i])
            if(boundaryAList[i]-boundaryAList[i-1])>800 and boundaryAList[i]>noisePoint:
                boundaryAIndex = i-1
                break
        
        boundaryA2=boundaryAList[boundaryAIndex]
        boundaryA = boundaryA2-100
        # plt.axvline(boundaryA2)
        # plt.axvline(boundaryA)
        # print("boundaryA,boundaryA2",boundaryA,boundaryA2)

        # A点
        curMin = dfData[boundaryA]
        curMinIndex = boundaryA
        for i in range(boundaryA,boundaryA2):
            if dfData[i]<curMin:
                curMinIndex=i
                curMin = dfData[i]
        a = curMinIndex
     

        df0 = self.df.copy()
        dfIndex = np.arange(0,len(df0),1)

        df1 = self.df.copy()
        df1 = df1[10:] # 去掉初始噪声
        df1['data_rolling'] = df1['data'].rolling(window=10).sum()# rolling
        
        
        tmp_a = df1[abs(df1.data_rolling) < 1].index.to_list()[0]
        # # 从temp_a点之后
        df1 = df1[tmp_a:]
        
        ###############定B点####################
        df2 = df1.copy()
        df2 = df2.loc[a:]
        tmp_b = df2[abs(df2.data_rolling - 10) < 0.7 ].index.to_list()
        tmp_b1 = df2[df2['data'] <= -1].index.to_list()[0] 
        df2 = df2.loc[tmp_b[0]+500:]
        df2_tmp = df2.loc[:tmp_b1]
        b = df2_tmp[abs(df2_tmp.data_rolling - 10) < 0.2 ].index.to_list()[-1]
        # print("b = " + str(b))
        df2 = self.df.loc[b-50:]

        ##############定C点######################### 

        df3 = df2.copy()
        tmp_c = df2[df3['data'] <= -1].index.to_list()[0]
        # plt.axvline(tmp_c)
        df3 = df3.loc[tmp_c:]
        # print('df3:',df3)
        tmp_c = df3[abs(df3['data'] -1) < 0.1].index.to_list()
        df3Data = df3['data']
        # tmp_c3=df3Data[tmp_c[0]:tmp_c[-1]].argmax()
        curMax = df3Data[tmp_c[0]]
        curMaxIndex = tmp_c[0]
       
        cBoundary =500# 找c点的边界范围
        if len(tmp_c)<cBoundary:
            cBoundary=len(tmp_c)-1
        # for i in range(1,len(tmp_c)):
        #     if(tmp_c[i]-tmp_c[i-1]>500):
        #         cBoundary=i-1
        #         break
        secondCorrectionPoint =tmp_c[0]
        for i in range(tmp_c[0],tmp_c[cBoundary]):
            if df3Data[i]>curMax:
                curMaxIndex=i
                curMax = df3Data[i]
            if df3Data[i]==1:
                secondCorrectionPoint=i
                

        c = curMaxIndex

        if(dfData[c]-dfData[tmp_c[0]]<0.05):
            c=tmp_c[0]

        # print("res_c:",c)

        #获得a,b,c三点
        # self.pointA = 2151
        # # self.pointA = 1972
        # self.pointB = 4519+self.pointA-1000
        # self.pointC = 7538+self.pointA-1000
        
        # pointA:2151 2171,pointB: 5670 5668, pointC: 8689,8449
        # a = 2151
        # b = 5670
        # c = 8689
        self.pointA = a
        self.pointB = b
        self.pointC = c
        self.secondCorrectPoint =secondCorrectionPoint
        # data_a_b_c = pd.read_csv('trian_data.csv')
        # fileNameAndType= 'fileNameAndType'
        
        # dict = {'fileNameAndType':[self.fileNameAndType],'a':[self.pointA], 'b':[self.pointB], 'c':[self.pointC]}
        # print(self.fileNameAndType,self.pointA,self.pointB,self.pointC)
        # dict = {''}
        # data_frame_a_b_c = pd.DataFrame(dict)
        # data_frame_a_b_c.to_csv('train_data.csv',index =False, header=False, mode='a')

        # print("a,b,c",a,b,c,secondCorrectionPoint)
        # plt.plot(df1.data)
    
      

    def set_pointABC(self,pointA,pointB,pointC):
        """
        set pointA,pointB,pointC
        """
        self.pointA=pointA
        self.pointB=pointB
        self.pointC = pointC
    
    # 曲线拟合

    def fit_tail(self,fitParameters):
        # print("name:",self.fileName)
        # plt.plot(self.df.data)
        # 获取拟合起始点和终止点
        tmpDf=self.df.data.copy()

        minPoint= tmpDf[100:].argmin()+100
        startP =minPoint+200
        for i in range(minPoint+1,len(tmpDf)):
            if(abs(tmpDf[i]-tmpDf[i-1])>0.001):
                # startP=i
                break
            startP+=1
        if startP==len(tmpDf):
            return 
        endP = startP
        startP-=40
        for i in range(startP+1,len(tmpDf)):
            if(abs(tmpDf[i]-tmpDf[i-1])<0.0001) and abs(tmpDf[i]-1)<0.05:
                # endP= i
                break
            endP+=1

        endP+=50
        if(endP>len(tmpDf)):
            return
        maxIndex = tmpDf[startP:endP].argmax()+startP
        startP=maxIndex-300
        endP=maxIndex+700
        plt.axvline(startP)
        plt.axvline(endP)
        dfData= self.df.data.copy()
        xList = [i for i in range(startP,endP)]
        # 拟合，参数10 是拟合目标n次幂函数
        coef = np.polyfit(xList,dfData[startP:endP],fitParameters)
        y_fit = np.polyval(coef,xList)
        # dfData[startP:endP]= y_fit.copy()

        self.fitdf= self.df.copy()
        # 拟合后的数据
        # self.fitData = dfData.copy()
        self.fitdf.data[startP:endP]=y_fit.copy()
  
    def get_df_ACor_bc(self, params = [1001, 11799]):
        #########获得修正后的曲线ACor#############

        # self.dT = (self.df['tick'][1] - self.df['tick'][0]) / 1000 # 采样频率
        # self.x = 1
        A = self.df['data'].copy()
        A = A[self.pointA-params[0]:self.pointA+params[1]] # 处理数据长度，零线长度1000，后续11799，可自行修改
        self.A = A
        fcut = 1000  # 低通滤波的截止频率
        forder = 3  # Butterworth低通滤波器的阶数
        wn = fcut * 2 * self.dT
        butterB, butterA = butter(forder, wn, 'low')
        A = lfilter(butterB, butterA, A)
        # A = signal.lfilter(butterB, butterA, A)  # matlab. filter()

        # 零线修正
        # self.zeroLinelength = 1000
        self.zeroLinelength = params[0]-1
        # A[:zeroLinelength].mean()
        # self.A
        self.ACor = A - A[:self.zeroLinelength].mean()

        #########从ACor中截取BC段##################
        # 起始点，终止点index：B点,C点的index 减去 曲线截去的zeroLinelength
        
        self.startPoint = self.pointB - (self.pointA-self.zeroLinelength)
        self.endPoint = self.pointC-(self.pointA-self.zeroLinelength)

        # 经过修正后的bc段
        self.dfACorBC = self.ACor[self.startPoint-1:self.endPoint]
        
    

    def make_smooth(self,smoothIter=500):
        """
        Attributes:
            smooth_iter: 平滑迭代次数
        """

        #平滑处理
        # self.dfACorBC
        minAocBC = min(self.dfACorBC)
        self.dfSmooth = self.dfACorBC.copy()
        
        

        for i in range(smoothIter):
            self.dfSmooth = smooth(self.dfSmooth,5)
            # self.dfSmooth = smooth(self.dfSmooth,19)
        minSmooth = min(self.dfSmooth)
        self.smoothErrorValue =abs((minAocBC - minSmooth)/minAocBC)
        print(self.dfSmooth)


    def get_sugget_smooth(self):
        """自动找到符合的smooth迭代次数
            Attributes:
        """
        self.dfSmooth = self.dfACorBC.copy()
        for i in range(100):
            self.dfSmooth = smooth(self.dfSmooth,5)
            # self.dfSmooth = smooth(self.dfSmooth,19)
        minSmooth = min(self.dfACorBC)
        minAocBC = min(self.dfACorBC)
        i=100
        errorValue = abs((minSmooth-minAocBC)/minAocBC)
        errorValue1 = round(errorValue,4)
        while(i<1000):
            minSmooth = min(self.dfSmooth)
            errorValue = abs((minSmooth-minAocBC)/minAocBC)
            if(errorValue>0.02): #控制在峰值2%误差内
                break
            i+=1
            # self.dfSmooth = smooth(self.dfSmooth,19)
            self.dfSmooth = smooth(self.dfSmooth,5)
        self.suggetIter=i
        # print(r"推荐平滑次数 (100 - " + str(self.suggetIter) + r")",i)
        # print("range from " + str(errorValue1 * 100) + "% to " + str(round(errorValue * 100,4)) + "%")
        self.smoothErrorValue = errorValue

    # def set_isCorrect(self,isCorrect):
    #     self.isCorrect=isCorrect

    def set_parameters(self,x,g,AT,H,M,deviceN,isCorrect):
        self.x = x
        self.g=g
        self.AT = AT
        self.H = H
        self.M=M
        self.deviceN=deviceN
        self.isCorrect=isCorrect
        

    def compute_formual(self):
        """
        根据平滑后的曲线计算公式
            Attributes:
                self.N: len(peimitiveCurve)

        """
        """
            x= 1 # 数据系统灵敏度， 单位mV/g
            # 处理数据长度
            # N = len(A_Cor)
            # 输入实验参数
            g = 9.80665
            # AT = eval(input("请输入温度:"))
            AT = 27
            # H = eval(input("请输入落锤高度(cm): "))
            H = 0.5
            # M = eval(input("请输入落锤质量(kg): "))
            M = 5140
            
            # n支试件
            deviceN=2
        """
        x= self.A
        g= self.g
        AT= self.AT
        H = self.H
        M =  self.M
        deviceN = self.deviceN
        
        # x= 1 # 数据系统灵敏度， 单位mV/g
        # # 处理数据长度
        # # N = len(A_Cor)
        # # 输入实验参数
        # g = 9.80665
        # # AT = eval(input("请输入温度:"))
        # AT = 27
        # # H = eval(input("请输入落锤高度(cm): "))
        # H = 0.5
        # # M = eval(input("请输入落锤质量(kg): "))
        # M = 5140
        
        # # n支试件
        # deviceN=2

        # self.x = 1 # 数据系统灵敏度，单位mV/g
        # self.g = 9.80665
        # # AT = eval(input("请输入温度:"))
        # self.AT = 27
        # # H = eval(input("请输入落锤高度(cm): "))
        # self.H = 0.5
        # # M = eval(input("请输入落锤质量(kg): "))
        # self.M = 5140
        self.X = []
        for i in range(15):
            self.X.append(0)

        self.X[11]=H
        self.X[12]=M
        self.X[13]=AT

        
        M = M/deviceN  # 使用两只试件同时开展试验时需要除以2
        

        # 读取文件
        self.X[14] = 1/x

        # t = [dT * i for i in range(len(A_Cor))]

        # 修正后的原始曲线，用于画图
        self.primitiveCurve=self.ACor.copy()
        self.primitiveCurve[:1000]=0
        

        
        ACor1 = self.ACor.copy()
        ACor1[self.startPoint-1:self.endPoint] = self.dfSmooth

        ACor1 = ACor1*g

        # 速度数据
        if H < 10:
            VStartPoint = self.zeroLinelength
            V0 = (ACor1[VStartPoint-1:self.startPoint].sum()-(ACor1[VStartPoint-1]+ACor1[self.startPoint-1])/2)*self.dT
            print('低落高冲击，由加速度积分得到初速度V0(m/s): %f' % V0)
        else:
            V0 = (2*g*H/100)**0.5
            print('低落高冲击，由加速度积分得到初速度V0(m/s): %f' % V0)
        #if str.lower(input('速度输入是否需要修正？(Y/N)')) == 'y':
        #    V0 = float(input('输入速度(m/s):')) 
            
        ACor1= ACor1[self.startPoint-1:self.endPoint]
        


        # t1 = np.array(range(len(ACor1)))*dT

        # 二次修正，这里也要设textBox
        #ppp4 = int(input('请输入二次修正点A2: '))
        
        ppp4 = self.secondCorrectPoint # input 输入
        # plt.plot(ACor1)
        # plt.show()
        # ppp4 = np.argwhere(ACor1==0.5)
        # ppp4=ACor1.argwhere(g)
        if self.isCorrect:
            # print("ppp4:",ppp4)
            AA = ACor1[:ppp4]
            ppp1 = V0 + (AA.cumsum() - AA[0] / 2 - AA / 2) * self.dT
            ppp2 = (ppp1.cumsum(0) - ppp1[0] / 2 - ppp1 / 2) * self.dT
            NN = len(AA)
            r1 = (2 * ppp2[NN - 1] / (NN * self.dT) ** 2)
            ACor1 = ACor1-r1
        V = V0 + (ACor1.cumsum() - (ACor1[0]/2) - (ACor1/2))*self.dT
        D = (V.cumsum() - (V[0]/2) - (V/2))*self.dT
        V[0] = V0
        D[0] = 0
        self.X[1] = V0

        if abs(D[:len(D) - 1].min()) >= abs(D[:len(D) - 1].max()):
            Dm = D[:len(D) - 1].min()
            indexDm = D[:len(D) - 1].argmin()
        else:
            Dm = D[:len(D) - 1].max()
            indexDm = D[:len(D) - 1].argmax()
        print('最大位移(mm): %f' % (Dm * 1000))
        Dm = D.max()
        self.X[2] = Dm

        Am = ACor1.min()
        indexAm = ACor1.argmin()
        print('最大加速度(g): %f' % (Am /g))
        self.X[0] = Am / g
        # print(len(ACor1))
        ii = indexDm
        while ii < len(ACor1) - 1:
            if (ACor1[ii] - g) * (ACor1[ii + 1] - g) < 0:
                break
            ii += 1
        
        indexDr = ii
        Dr = D[indexDr]
        print('残余变形量(mm): %f' % (Dr * 1000))

        ii = indexDm
        while ii < len(ACor1) - 1:
            if D[ii] < 0:
                break
            ii += 1
        indexD0 = ii

        # indexD0
        if indexDr <= indexD0:
            indexD0 = indexDr
            
        # 计算力
        F = -M * (ACor1 - g)
        F = F / 1000
        Fm = abs(F).max()
        indexFm = abs(F).argmax()
        print('最大的抗力(kN): %f' % Fm)
        self.X[3] = Fm

        FDm = F[indexDm]
        print('位移最大时刻的抗力(kN): %f' % FDm)

        # 计算能量
        temp1 = F * D
        temp2 = F[1:] * D[:len(D) - 1]
        temp3 = F[:len(F) - 1] * D[1:len(D)]

        # 最大冲击能量
        Wm = (temp1[1:indexDm + 1] - temp2[:indexDm] + temp3[:indexDm] - temp1[:indexDm]).sum()/2
        print('最大冲击能量Wm(kJ) %f ' % (Wm / 1000))
        self.X[4] = Wm

        # 剩余冲击能量
        Wr = (temp1[indexDm + 1:indexDr + 1] - temp2[indexDm:indexDr] + temp3[indexDm:indexDr] - temp1[
                                                                                                    indexDm:indexDr]).sum() / (
                -2)
        print('剩余冲击能量Wr(kJ) %f ' % (Wr / 1000))

        # 缓冲器吸收的能量
        Wa = Wm - Wr
        print('缓冲器吸收的能量Wa(kJ) %f' % (Wa / 1000))
        self.X[5] = Wa
        SF = Wa / Wm
        self.X[9] = SF

            
        Kp = abs(FDm / Dm)
        self.X[6] = Kp
        Ke = 2 * Wm / (Dm ** 2)
        self.X[7] = Ke
        c = (Wa - Kp * (Dr ** 2) / 2) * 1000 / (
                    (np.power(V[:indexDr+1], 2).sum() - (np.power(V[0], 2) + np.power(V[indexDr], 2)) / 2) * self.dT)
        self.X[8] = c
        n = c / (2 * np.power((M * Kp * 1000), 0.5))
        self.X[10] = n
        

        # ACor1：对修正后的曲线ACor 进行*g等计算处理后的曲线
        self.ACor1 = ACor1
        self.V = V
        self.D = D
        self.N = len(self.primitiveCurve)
        self.g = g
        self.F = F


        #  计算得到的结果数值
        x = self.X
        Am = x[0]
        Vo = x[1]
        Dm = x[2]
        Fm = x[3]
        Wm = x[4]
        Wa = x[5]
        Kp = x[6]
        Ke = x[7]
        self.C = round(x[8], 2)
        SF = x[9]
        n = x[10]
        self.H = x[11]
        self.M = x[12]
        #At = [num2str(x(:, 14)) '℃']
        self.At = x[13]
        self.Am = -(round(Am * 100)) / 100
        self.Vo = (round(Vo * 1000)) / 1000
        self.Dm = round(Dm * 100000) / 100
        self.Fm = round(Fm * 100) / 100
        self.Wm = round(Wm * 1000)
        self.Wa = round(Wa * 1000)
        self.Kp = round(Kp * 10) / 10
        # print("Kp %f" % self.Kp)
        self.Ke = round(Ke * 10) / 10
        self.SF = (round(SF * 1000)) / 1000
        self.n = (round(n * 1000)) / 1000



        Am = -(round(Am * 100)) / 100
        Vo = (round(Vo * 1000)) / 1000
        Dm = round(Dm * 100000) / 100
        Fm = round(Fm * 100) / 100
        Wm = round(Wm * 1000)
        Wa = round(Wa * 1000)
        Kp = round(Kp * 10) / 10
        Ke = round(Ke * 10) / 10
        SF = (round(SF * 1000)) / 1000
        n = (round(n * 1000)) / 1000



        #  计算表格中的值（位移，抗力，分段刚度，上宽，下宽）
        f= self.F.copy()
        d = self.D.copy() *1000
        v = self.V.copy()

        N3 = len(d)
        N4 = len(f)
        if N3 < N4:
            d=d[0:N3]
            f = f[0:N3]
            # f00 = (f[N3-1] + f[0]) / 2
        else:
            d = d[0:N4]
            f = f[0:N4]
            # f00 = (f[N4-1] + f[0]) / 2

        #-----绘制十等分曲线及计算分段刚度,上宽,下宽等
        DD = [0 for i in range(11)]
        # DD[0]=0
        for jj in range(1,11):
            DD[jj] = Dm/10+DD[jj-1]
            DD[jj] = round(DD[jj]*1000)/1000
        DD = np.array(DD)
        


        dm = d.max()
        xj = d.argmax()
        qd = d[:xj+1]
        j1 = 0
        j2 =np.argwhere(qd>=abs(DD[1])) 
        j2 = j2.min()
        j3 = np.argwhere(qd>=abs(DD[2]))
        j3 = j3.min()
        j4 = np.argwhere(qd>=abs(DD[3]))
        j4 = j4.min()
        j5 = np.argwhere(qd>=abs(DD[4]))
        j5 = j5.min()
        j6 = np.argwhere(qd>=abs(DD[5]))
        j6 = j6.min()
        j7 = np.argwhere(qd>=abs(DD[6]))
        j7 = j7.min()
        j8 = np.argwhere(qd>=abs(DD[7]))
        j8 = j8.min()
        j9 = np.argwhere(qd>=abs(DD[8]))
        j9 = j9.min()
        j10 = np.argwhere(qd>=abs(DD[9]))
        j10 = j10.min()

        hd = d[xj:len(d)]
        d1 = np.array([abs(d[kk]) for kk in range(len(d))])

        d1m=d1[xj:len(d)].min()
        yj = d1[xj:len(d)].argmin()
        jj1 = yj+xj
        jj2 = np.argwhere(hd>=abs(DD[1]))
        jj2 = jj2.max()+xj
        jj3 = np.argwhere(hd>=abs(DD[2]))
        jj3 = jj3.max()+xj
        jj4 = np.argwhere(hd>=abs(DD[3]))
        jj4 = jj4.max()+xj
        jj5 = np.argwhere(hd>=abs(DD[4]))
        jj5 = jj5.max()+xj
        jj6 = np.argwhere(hd>=abs(DD[5]))
        jj6 = jj6.max()+xj
        jj7 = np.argwhere(hd>=abs(DD[6]))
        jj7 = jj7.max()+xj
        jj8 = np.argwhere(hd>=abs(DD[7]))
        jj8 = jj8.max()+xj
        jj9 = np.argwhere(hd>=abs(DD[8]))
        jj9 = jj9.max()+xj
        jj10 = np.argwhere(hd>=abs(DD[9]))
        jj10 = jj10.max()+xj

        F1= 0
        # F1 = (f[j1]*abs(v[jj1])+f[jj1]*abs(v[j1]))/(abs(v[j1])+abs(v[jj1]))
        F2 = (f[j2]*abs(v[jj2])+f[jj2]*abs(v[j2]))/(abs(v[j2])+abs(v[jj2]))
        F3 = (f[j3]*abs(v[jj3])+f[jj3]*abs(v[j3]))/(abs(v[j3])+abs(v[jj3]))
        F4 = (f[j4]*abs(v[jj4])+f[jj4]*abs(v[j4]))/(abs(v[j4])+abs(v[jj4]))
        F5 = (f[j5]*abs(v[jj5])+f[jj5]*abs(v[j5]))/(abs(v[j5])+abs(v[jj5]))
        F6 = (f[j6]*abs(v[jj6])+f[jj6]*abs(v[j6]))/(abs(v[j6])+abs(v[jj6]))
        F7 = (f[j7]*abs(v[jj7])+f[jj7]*abs(v[j7]))/(abs(v[j7])+abs(v[jj7]))
        F8 = (f[j8]*abs(v[jj8])+f[jj8]*abs(v[j8]))/(abs(v[j8])+abs(v[jj8]))
        F9 = (f[j9]*abs(v[jj9])+f[jj9]*abs(v[j9]))/(abs(v[j9])+abs(v[jj9]))
        F10 = (f[j10]*abs(v[jj10])+f[jj10]*abs(v[j10]))/(abs(v[j10])+abs(v[jj10]))
        F =np.array([F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,f[xj]])
        for i in range(len(F)):
            F[i]=round(F[i]*1000)/1000



        KK1 = 0
        KK2 = (F2-F1)/(DD[1]-DD[0])
        KK3 = (F3-F2)/(DD[2]-DD[1])
        KK4 = (F4-F3)/(DD[3]-DD[2])
        KK5 = (F5-F4)/(DD[4]-DD[3])
        KK6 = (F6-F5)/(DD[5]-DD[4])
        KK7 = (F7-F6)/(DD[6]-DD[5])
        KK8 = (F8-F7)/(DD[7]-DD[6])
        KK9 = (F9-F8)/(DD[8]-DD[7])
        KK10 = (F10-F9)/(DD[9]-DD[8])
        KK11 = (f[xj]-F10)/(DD[10]-DD[9])
        KK = np.array([KK1,KK2,KK3,KK4,KK5,KK6,KK7,KK8,KK9,KK10,KK11])
        for i in range(len(KK)):
            KK[i]=round(KK[i]*10000)/10


        FF1 = abs(f[j1]-f[jj1])*abs(v[j1])/(abs(v[j1])+abs(v[jj1]))
        FFF1 = abs(f[j1]-f[jj1])*abs(v[jj1])/(abs(v[j1])+abs(v[jj1]))

        FF2 = abs(f[j2]-f[jj2])*abs(v[j2])/(abs(v[j2])+abs(v[jj2]))
        FFF2 = abs(f[j2]-f[jj2])*abs(v[jj2])/(abs(v[j2])+abs(v[jj2]))
        FF3 = abs(f[j3]-f[jj3])*abs(v[j3])/(abs(v[j3])+abs(v[jj3]))
        FFF3 = abs(f[j3]-f[jj3])*abs(v[jj3])/(abs(v[j3])+abs(v[jj3]))
        FF4 = abs(f[j4]-f[jj4])*abs(v[j4])/(abs(v[j4])+abs(v[jj4]))
        FFF4 = abs(f[j4]-f[jj4])*abs(v[jj4])/(abs(v[j4])+abs(v[jj4]))
        FF5 = abs(f[j5]-f[jj5])*abs(v[j5])/(abs(v[j5])+abs(v[jj5]))
        FFF5 = abs(f[j5]-f[jj5])*abs(v[jj5])/(abs(v[j5])+abs(v[jj5]))
        FF6 = abs(f[j6]-f[jj6])*abs(v[j6])/(abs(v[j6])+abs(v[jj6]))
        FFF6 = abs(f[j6]-f[jj6])*abs(v[jj6])/(abs(v[j6])+abs(v[jj6]))
        FF7 = abs(f[j7]-f[jj7])*abs(v[j7])/(abs(v[j7])+abs(v[jj7]))
        FFF7 = abs(f[j7]-f[jj7])*abs(v[jj7])/(abs(v[j7])+abs(v[jj7]))
        FF8 = abs(f[j8]-f[jj8])*abs(v[j8])/(abs(v[j8])+abs(v[jj8]))
        FFF8 = abs(f[j8]-f[jj8])*abs(v[jj8])/(abs(v[j8])+abs(v[jj8]))
        FF9 = abs(f[j9]-f[jj9])*abs(v[j9])/(abs(v[j9])+abs(v[jj9]))
        FFF9 = abs(f[j9]-f[jj9])*abs(v[jj9])/(abs(v[j9])+abs(v[jj9]))
        FF10 = abs(f[j10]-f[jj10])*abs(v[j10])/(abs(v[j10])+abs(v[jj10]))
        FFF10 = abs(f[j10]-f[jj10])*abs(v[jj10])/(abs(v[j10])+abs(v[jj10]))

        FF = [FF1,FF2,FF3,FF4,FF5,FF6,FF7,FF8,FF9,FF10,0]
        FFF = [FFF1,FFF2,FFF3,FFF4,FFF5,FFF6,FFF7,FFF8,FFF9,FFF10,0]
        # 数值位数转换
        for i in range(len(DD)):
            DD[i]=round(DD[i]*100)/100
            F[i] = round(F[i]*100)/100
            FF[i] = abs(round(FF[i]*100)/100)
            FFF[i] = abs(round(FFF[i]*100)/100)

        # print(DD,F,FFF)

        ##########右下角表格绘制##########

        F1= np.c_[DD,F,KK,FF,FFF]
        self.F1 = F1.copy()

        # print(F1,F1.shape)
        # # plt.rcParams['font.family']=['STFangsong'] #设置字体，否则中文会显示为方块
        # plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
        # plt.rcParams["axes.unicode_minus"]=False #正常显示负号
        # col=['位移(mm)', '抗力(kN)', '分段刚度(kN/m)', '上宽(kN)', '下宽(kN)']
        # plt.table(cellText=F1,
        #         colLabels=col,
        #         loc='center',
        #         cellLoc='center',
        #         rowLoc='center')
        # plt.title('10等分位移－抗力对应值')
        # plt.axis('off')
        # plt.show()





    def get_result(self):
        x = self.X
        Am = x[0]
        Vo = x[1]
        Dm = x[2]
        Fm = x[3]
        Wm = x[4]
        Wa = x[5]
        Kp = x[6]
        Ke = x[7]
        self.C = round(x[8], 2)
        SF = x[9]
        n = x[10]
        self.H = x[11]
        self.M = x[12]
        #At = [num2str(x(:, 14)) '℃']
        self.At = x[13]
        self.Am = -(round(Am * 100)) / 100
        self.Vo = (round(Vo * 1000)) / 1000
        self.Dm = round(Dm * 100000) / 100
        self.Fm = round(Fm * 100) / 100
        self.Wm = round(Wm * 1000)
        self.Wa = round(Wa * 1000)
        self.Kp = round(Kp * 10) / 10
        self.Ke = round(Ke * 10) / 10
        self.SF = (round(SF * 1000)) / 1000
        self.n = (round(n * 1000)) / 1000

   

        
       
# import os

# 读取所有.dat文件
# filePath = "./data"
# dataFile = os.listdir(filePath)
# print(dataFile[0].split(".")[0])
# print(dataFile)
# process = ProcessCurve()
# # print("./data/"+dataFile[0])

# # process.set_df2('20100125-17A.dat',"20100125-17A")
# # process.determine_a_b_c()

# for i in range(len(dataFile)):
#     process.set_df("./data/"+dataFile[i])
#     process.determine_a_b_c()

# process = ProcessCurve()
# dataFile = "73mm1.mat"
# dataFile2="20100125-17A.dat"
# dataFile3= "20100324-5A.dat"

# process = ProcessCurve()
# process.train_a()


# process.set_df(dataFile2)
# process.model_predict_a_b_c()
# # process.dataInit("./data/"+dataFile3)
# # process.dataInit("./"+dataFile2)
# # process.set_df("./"+dataFile)
# process.determine_a_b_c()


# process.get_df_ACor_bc()
# process.make_smooth(500)
# process.get_sugget_smooth()
# x= 1 # 数据系统灵敏度， 单位mV/g
# 处理数据长度
# N = len(A_Cor)
# 输入实验参数
# g = 9.80665
# AT = eval(input("请输入温度:"))
# AT = 27
# H = eval(input("请输入落锤高度(cm): "))
# H = 0.5
# M = eval(input("请输入落锤质量(kg): "))
# M = 5140

# n支试件
# deviceN=2
# process.compute_formual(x,g,AT,H,M,deviceN)
# print(process.X)
