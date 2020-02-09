import sys
from PyQt5 import QtWidgets
import gui


class CycleGan(QtWidgets.QMainWindow, gui.Ui_MainWindow):
    dataTrainPath = ""
    dataGenPath = ""
    XtoYPath = ""
    YtoXPath = ""
    imsizeTrain = 512
    imsizeGen = 512
    lr = 0.0001
    epochs = 30
    trainStatus = ""
    trainProgressAmnt = 0
    genProgressAmnt = 0


    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.lrslider.valueChanged.connect(self.updateAll)
        self.sizetrainslider.valueChanged.connect(self.updateAll)
        self.sizegenslider.valueChanged.connect(self.updateAll)
        self.epochsslider.valueChanged.connect(self.updateAll)
        self.datatrainselect.clicked.connect(self.select_dataTrainPath)
        self.datagenselect.clicked.connect(self.select_dataGenPath)
        self.xtoyselectbtn.clicked.connect(self.select_XtoY)
        self.ytoxselectbtn.clicked.connect(self.select_YtoX)
        self.starttrainbtn.clicked.connect(self.startTrain)
        self.startgenbtn.clicked.connect(self.startGen)
        self.reset.clicked.connect(self.setlol)
        self.updateAll()



    def select_dataTrainPath(self):
        self.dataTrainPath = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку датасета")
        if not self.dataTrainPath:
            self.showError("Ошибка", "Папка не выбрана", "Выберите папку")

    def select_dataGenPath(self):
        self.dataGenPath = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку датасета")
        if not self.dataGenPath:
            self.showError("Ошибка", "Папка не выбрана", "Выберите папку")

    def select_XtoY(self):
        self.XtoYPath = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл модели XtoY")[0]
        print(self.XtoYPath)
        if not self.XtoYPath:
            self.showError("Ошибка", "Файл не выбран", "Выберите файл")

    def select_YtoX(self):
        self.YtoXPath = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл YtoX")[0]
        print(self.YtoXPath)
        if not self.YtoXPath:
            self.showError("Ошибка", "Файл не выбран", "Выберите файл")

    def showError(self, error, text, inftext):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setWindowTitle(error)
        msg.setText(text)
        msg.setInformativeText(inftext)
        msg.exec_()

    def updateVars(self):
        self.imsizeTrain = (self.sizetrainslider.value()//32)*32
        self.imsizeGen = (self.sizegenslider.value()//32)*32
        self.lr = self.lrslider.value()/1000
        self.epochs = self.epochsslider.value()
        print(self.imsizeTrain, self.imsizeGen, self.lr, self.epochs)

    def updateLbls(self):
        self.lramnt.setText(str(self.lr))
        self.epochsamnt.setText(str(self.epochs))
        self.sizetrainamnt.setText(str(self.imsizeTrain) + "px")
        self.sizegenamnt.setText(str(self.imsizeGen) + "px")

    def updateAll(self):
        self.updateVars()
        self.updateLbls()
        self.updateStatuses()

    def updateStatuses(self):
        self.trainProgress.setValue(self.trainProgressAmnt)
        self.genProgress.setValue(self.genProgressAmnt)
        print(self.trainProgress.value())

    def setStatuses(self, trainEpoch = 0, genEpoch = 0, genAll = 1):
        self.trainProgressAmnt = (trainEpoch/self.epochs)*100
        self.genProgressAmnt = (genEpoch/genAll)*100
        self.updateStatuses()
        #print(trainEpoch, self.trainProgressAmnt)

    def startTrain(self):
        if not self.dataTrainPath:
            self.showError("Ошибка", "Нет датасета", "Выберите папку с датасетом")
            self.select_dataTrainPath()
        else:
            from trainfunc import train
            train(epochs=self.epochs, dataset = self.dataTrainPath, lr=self.lr, imSize=self.imsizeTrain, cuda=True)

    def startGen(self):
        from testfunc import test
        if not self.dataGenPath:
            self.showError("Ошибка", "Нет датасета", "Выберите папку с датасетом")
            self.select_dataGenPath()
        elif not self.XtoYPath:
            self.showError("Ошибка", "Нет Модели", "Выберите файл модели XtoY")
            self.select_XtoY()
        elif not self.YtoXPath:
            self.showError("Ошибка", "Нет Модели", "Выберите файл модели YtoX")
            self.select_YtoX()
        else:
            test(dataset=self.dataGenPath, imSize = self.imsizeGen, genXtoY = self.XtoYPath, genYtoX = self.YtoXPath)
    def setlol(self):
        self.trainProgress.setValue(self.trainProgress.value() + 10)

app = QtWidgets.QApplication(sys.argv)
window = CycleGan()


def main():
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
