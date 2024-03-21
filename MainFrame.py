
from MainUI import Ui_MainWindow
from PyQt5.QtWidgets import QWidget, QFileDialog, QMainWindow
from PyQt5 import QtWidgets
import sys
from dataProcessForm import *

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.dataProcessF = MyForm()
        self.horizontalLayout_2.addWidget(self.dataProcessF)
       
        self.pushButton.clicked.connect(self.makeButtonClicked)
        self.pushButton_2.clicked.connect(self.selectFileButtonClicked)
    
    def selectFileButtonClicked(self):
        # filePath =c.getOpenFileName(self,"打开",".","Text Files(*.txt *.dat *.csv *.mat);;All Files (*)")
        filePath, ok= QFileDialog.getOpenFileName(self,"打开",".","Text Files(*.txt *.dat *.csv *.mat);;All Files (*)")
        filelist = filePath.split('/')
        # os.sep：文件路径斜杠
        filePath=filelist[0]+os.sep
        for i in range(1,len(filelist)):
            filePath=os.path.join(filePath,filelist[i])
        print(filePath)
        self.lineEdit.setText(filePath)


    def makeButtonClicked(self):
        # self.splitter
        # self.dataProcessF.set_file(self.lineEdit.text())
        self.dataProcessF.processInit(self.lineEdit.text())
    
    # def emitPushButtonClicked(self):
    #     self.dict = {"path":"D:/智能化数据处理软件项目/20100125-17A.dat"}
    #     self.refresh_analyseData_signal.emit(self.dict)

    # def mySlot(self,dict):
    #     self.lineEdit.setText(dict["path"])
        
    
if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    # MainWindow = QtWidgets.QMainWindow()
    myWin = MyWindow()
    myWin.show()
    # myWin.setupUi(MainWindow)
    # MainWindow.show()
    sys.exit(app.exec_())