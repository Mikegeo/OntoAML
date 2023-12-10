# -*- coding: utf-8 -*-

# Form implementation generated from reading training_ui file 'main_page.training_ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtWidgets
from main.create.create_main import Ui_Training
from main.predict.predict_main import Ui_prediction as Form1


class Ui_MainWindow(object):

    def __init__(self):
        super().__init__()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(439, 292)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.make_Pipelines_btn = QtWidgets.QPushButton(self.centralwidget)
        self.make_Pipelines_btn.setGeometry(QtCore.QRect(50, 150, 141, 41))
        self.make_Pipelines_btn.setObjectName("make_Pipelines_btn")
        self.make_Predictions_btn = QtWidgets.QPushButton(self.centralwidget)
        self.make_Predictions_btn.setGeometry(QtCore.QRect(240, 150, 141, 41))
        self.make_Predictions_btn.setObjectName("make_Predictions_btn")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 40, 261, 61))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 439, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # #############################################
        # ############ connect buttons ################
        # #############################################

        self.make_Pipelines_btn.clicked.connect(self.openTraining)
        self.make_Predictions_btn.clicked.connect(self.openPrediction)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.make_Pipelines_btn.setText(_translate("MainWindow", "Create Pipelines"))
        self.make_Predictions_btn.setText(_translate("MainWindow", "Make Predictions"))
        self.label.setText(_translate("MainWindow", "AUTOMATED MACHINE LEARNING"))

    def openTraining(self):
        training = QtWidgets.QDialog()
        training.ui = Ui_Training()
        training.ui.setupUi(training)
        training.exec_()

    def openPrediction(self):
        prediction = QtWidgets.QDialog()
        prediction.ui = Form1()
        prediction.ui.setupUi(prediction)
        prediction.exec_()
        prediction.show()


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui1 = Ui_MainWindow()
    ui1.setupUi(MainWindow)
    training = QtWidgets.QDialog()
    training.ui = Ui_Training()
    training.ui.setupUi(training)
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
