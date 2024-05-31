# -*- coding: utf-8 -*-

# Form implementation generated from reading training_ui file 'Try10.training_ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import json
from owlready2 import *
from main.conf_files import conf
import numpy as np
import sys

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from main.create.data_analyzer import DataAnalyzer
from main.create.data_profiler import DataProfiler
from main.enums.python_pipeline_enums import PythonPipelineEnums, PipelineCodeWithMetricsEnum
from main.predict.predict_main import Ui_prediction as Form1

import owlready2
owlready2.JAVA_EXE = r"C:\Users\michalis.g\Protege-5.5.0\jre\bin\java.exe"


class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super(CheckableComboBox, self).__init__(parent)
        self.view().pressed.connect(self.handle_item_pressed)
        self.setModel(QStandardItemModel(self))

    # when any item get pressed
    def handle_item_pressed(self, index):
        # getting which item is pressed
        item = self.model().itemFromIndex(index)
        # make it check if unchecked and vice-versa
        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
        else:
            item.setCheckState(Qt.Checked)
        # calling method
        self.check_items()

    # method called by check_items
    def item_checked(self, index):
        # getting item at index
        item = self.model().item(index, 0)
        # return true if checked else false
        return item.checkState() == Qt.Checked

    # calling method
    def check_items(self):
        # blank list
        checkedItems = []
        # traversing the items
        for i in range(self.count()):
            # if item is checked add it to the list
            if self.item_checked(i):
                checkedItems.append(i)
        # call this method
        self.update_labels(checkedItems)
        return checkedItems

    def update_labels(self, item_list):
        n = ''
        count = 0
        labels = []
        for i in item_list:
            text_label = self.model().item(i, 0).text()
            if count == 0:
                n += ' %s' % i
                labels.append(text_label)
            else:
                n += ', %s' % i
                labels.append(text_label)

            count += 1

        for i in range(self.count()):

            text_label = self.model().item(i, 0).text()

            #            print('Current (index %s) text_label "%s"' % (i, text_label))
            #            sys.stdout.flush()
            if text_label.find('-') >= 0:
                text_label = text_label.split('-')[0]

            item_new_text_label = text_label + ' - selected item(s): ' + n

            # item_new_text_label = text_label

            #            self.model().item(i).setText(item_new_text_label)
            self.setItemText(i, item_new_text_label)  # copy/paste error corrected.
            # training.training_ui.deleteColumnsText.setText(",".join(map(str, labels)))

    sys.stdout.flush()


class Ui_Training(object):

    def __init__(self):
        super(Ui_Training, self).__init__()
        self.data_profiler = DataProfiler(self)
        self.data_analyzer = DataAnalyzer(self)

    def setupUi(self, Training):
        Training.setObjectName("Training")
        Training.resize(636, 560)
        font = QtGui.QFont()
        font.setKerning(False)
        Training.setFont(font)
        Training.setMouseTracking(False)
        self.gridLayoutWidget_2 = QtWidgets.QWidget(Training)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(30, 90, 521, 83))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_Dataset = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_Dataset.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_Dataset.setObjectName("gridLayout_Dataset")
        self.newProjectsPathLabel = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.newProjectsPathLabel.setObjectName("newProjectsPathLabel")
        self.gridLayout_Dataset.addWidget(self.newProjectsPathLabel, 2, 1, 1, 1)
        self.datasetDetailsBtn = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.datasetDetailsBtn.setAutoDefault(False)
        self.datasetDetailsBtn.setEnabled(False)
        self.datasetDetailsBtn.setObjectName("datasetDetailsBtn")
        self.gridLayout_Dataset.addWidget(self.datasetDetailsBtn, 1, 0, 1, 1)
        self.datasetPathLabel = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.datasetPathLabel.setObjectName("datasetPathLabel")
        self.gridLayout_Dataset.addWidget(self.datasetPathLabel, 1, 1, 1, 1)
        self.openDatasetProfileBtn = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.openDatasetProfileBtn.setAutoDefault(False)
        self.openDatasetProfileBtn.setEnabled(False)
        self.openDatasetProfileBtn.setObjectName("openDatasetProfileBtn")
        self.gridLayout_Dataset.addWidget(self.openDatasetProfileBtn, 2, 0, 1, 1)
        self.uploadDatasetBtn = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.uploadDatasetBtn.setAutoDefault(False)
        self.uploadDatasetBtn.setObjectName("uploadDatasetBtn")
        self.gridLayout_Dataset.addWidget(self.uploadDatasetBtn, 0, 0, 1, 1)
        self.datasetPathLineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.datasetPathLineEdit.setObjectName("datasetPathLineEdit")
        self.datasetPathLineEdit.setReadOnly(True)
        self.gridLayout_Dataset.addWidget(self.datasetPathLineEdit, 1, 2, 1, 1)
        self.newProjectsPathLineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.newProjectsPathLineEdit.setObjectName("newProjectsPathLineEdit")
        self.newProjectsPathLineEdit.setReadOnly(True)
        self.gridLayout_Dataset.addWidget(self.newProjectsPathLineEdit, 2, 2, 1, 1)
        self.formLayoutWidget_2 = QtWidgets.QWidget(Training)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(30, 180, 131, 81))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.datasetDetailsLayout = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.datasetDetailsLayout.setContentsMargins(0, 0, 0, 0)
        self.datasetDetailsLayout.setHorizontalSpacing(1)
        self.datasetDetailsLayout.setObjectName("datasetDetailsLayout")
        self.textViewSampleSize = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.textViewSampleSize.setReadOnly(True)
        self.textViewSampleSize.setObjectName("textViewSampleSize")
        self.datasetDetailsLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.textViewSampleSize)
        self.textViewFeatureSize = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.textViewFeatureSize.setReadOnly(True)
        self.textViewFeatureSize.setObjectName("textViewFeatureSize")
        self.datasetDetailsLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.textViewFeatureSize)
        self.sampleSizeLabel = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.sampleSizeLabel.setObjectName("sampleSizeLabel")
        self.datasetDetailsLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.sampleSizeLabel)
        self.featureSizeLabel = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.featureSizeLabel.setObjectName("featureSizeLabel")
        self.datasetDetailsLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.featureSizeLabel)
        self.datasetDetailsLabel = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.datasetDetailsLabel.setObjectName("datasetDetailsLabel")
        self.datasetDetailsLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.datasetDetailsLabel)
        self.verticalLayoutWidget = QtWidgets.QWidget(Training)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(170, 180, 382, 241))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.userPrefLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.userPrefLayout.setContentsMargins(0, 0, 0, 0)
        self.userPrefLayout.setObjectName("userPrefLayout")
        self.userPrefGrpBox = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.userPrefGrpBox.setObjectName("userPrefGrpBox")
        self.selectLabelColumnLabel = QtWidgets.QLabel(self.userPrefGrpBox)
        self.selectLabelColumnLabel.setGeometry(QtCore.QRect(10, 130, 171, 16))
        self.selectLabelColumnLabel.setObjectName("selectLabelColumnLabel")
        self.HPSeachGrpBox = QtWidgets.QGroupBox(self.userPrefGrpBox)
        self.HPSeachGrpBox.setGeometry(QtCore.QRect(160, 20, 155, 61))
        self.HPSeachGrpBox.setObjectName("HPSeachGrpBox")
        self.rationButtonExtensiveSearch = QtWidgets.QRadioButton(self.HPSeachGrpBox)
        self.rationButtonExtensiveSearch.setGeometry(QtCore.QRect(10, 20, 121, 17))
        self.rationButtonExtensiveSearch.setObjectName("rationButtonExtensiveSearch")
        self.radioButtonRandomSearch = QtWidgets.QRadioButton(self.HPSeachGrpBox)
        self.radioButtonRandomSearch.setGeometry(QtCore.QRect(10, 40, 111, 17))
        self.radioButtonRandomSearch.setObjectName("radioButtonRandomSearch")
        self.enhaceModelGrpBox = QtWidgets.QGroupBox(self.userPrefGrpBox)
        self.enhaceModelGrpBox.setGeometry(QtCore.QRect(5, 20, 141, 91))
        self.enhaceModelGrpBox.setObjectName("enhaceModelGrpBox")
        self.radioButtonInterpretability = QtWidgets.QRadioButton(self.enhaceModelGrpBox)
        self.radioButtonInterpretability.setGeometry(QtCore.QRect(10, 20, 111, 17))
        self.radioButtonInterpretability.setObjectName("radioButtonInterpretability")
        self.radioButtonAccuracy = QtWidgets.QRadioButton(self.enhaceModelGrpBox)
        self.radioButtonAccuracy.setGeometry(QtCore.QRect(10, 40, 100, 17))
        self.radioButtonAccuracy.setObjectName("radioButtonAccuracy")
        self.radioButtonSpeed = QtWidgets.QRadioButton(self.enhaceModelGrpBox)
        self.radioButtonSpeed.setGeometry(QtCore.QRect(10, 60, 91, 17))
        self.radioButtonSpeed.setObjectName("radioButtonSpeed")
        self.deleteComboBoxLabel = QtWidgets.QLabel(self.userPrefGrpBox)
        self.deleteComboBoxLabel.setGeometry(QtCore.QRect(10, 190, 300, 21))
        self.deleteComboBoxLabel.setObjectName("deleteComboBoxLabel")
        self.comboBoxTargetColumn = QtWidgets.QComboBox(self.userPrefGrpBox)
        self.comboBoxTargetColumn.setGeometry(QtCore.QRect(10, 150, 211, 23))
        self.comboBoxTargetColumn.setEditable(False)
        self.comboBoxTargetColumn.setCurrentText("")
        self.comboBoxTargetColumn.setObjectName("comboBoxTargetColumn")
        self.comboBoxDeleteColumns = CheckableComboBox(self.userPrefGrpBox)
        self.comboBoxDeleteColumns.setGeometry(QtCore.QRect(10, 210, 211, 23))
        self.comboBoxDeleteColumns.setObjectName("comboBoxDeleteColumns")
        self.userPrefLayout.addWidget(self.userPrefGrpBox)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Training)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(170, 435, 249, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.buttonsLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.buttonsLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.buttonsLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonsLayout.setObjectName("buttonsLayout")
        self.runPipelinesBtn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.runPipelinesBtn.setAutoDefault(False)
        self.runPipelinesBtn.setObjectName("runPipelinesBtn")
        self.buttonsLayout.addWidget(self.runPipelinesBtn)
        self.createPipelinesBtn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.createPipelinesBtn.setAutoDefault(False)
        self.createPipelinesBtn.setObjectName("createPipelinesBtn")
        self.buttonsLayout.addWidget(self.createPipelinesBtn)
        self.backBtn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.backBtn.setAutoDefault(False)
        self.backBtn.setObjectName("backBtn")
        self.buttonsLayout.addWidget(self.backBtn)
        self.runPredictionsBtn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.runPredictionsBtn.setAutoDefault(False)
        self.runPredictionsBtn.setObjectName("runPredictionsBtn")
        self.runPredictionsBtn.setEnabled(False)
        self.buttonsLayout.addWidget(self.runPredictionsBtn)

        self.retranslateUi(Training)
        QtCore.QMetaObject.connectSlotsByName(Training)

        # #############################################
        # ############ connect buttons ################
        # #############################################

        self.uploadDatasetBtn.clicked.connect(self.uploadDataset)
        self.openDatasetProfileBtn.clicked.connect(self.openDatasetProfiling)
        self.createPipelinesBtn.clicked.connect(self.runReasoner)
        self.runPipelinesBtn.clicked.connect(self.runPipelines)
        # back button
        self.backBtn.clicked.connect(self.back)
        self.backBtn.clicked.connect(Training.close)
        self.runPredictionsBtn.clicked.connect(Training.close)
        self.runPredictionsBtn.clicked.connect(self.run_predictions)
        self.datasetDetailsBtn.clicked.connect(self.data_profiler.datasetDetailsProfiling)

    def retranslateUi(self, Training):
        _translate = QtCore.QCoreApplication.translate
        Training.setWindowTitle(_translate("Training", "Create Pipelines"))
        self.uploadDatasetBtn.setText(_translate("Training", "Upload Dataset"))
        self.datasetDetailsBtn.setText(_translate("Training", "Analyse Dataset"))
        self.openDatasetProfileBtn.setText(_translate("Training", "View Dataset Profile"))
        self.datasetPathLabel.setText(_translate("Training", "Dataset\'s Path"))
        self.newProjectsPathLabel.setText(_translate("Training", "Created Project\'s Path"))
        self.sampleSizeLabel.setText(_translate("Training", "Sample Size"))
        self.featureSizeLabel.setText(_translate("Training", "Feature Size"))
        self.datasetDetailsLabel.setText(_translate("Training", "Dataset\'s Details"))
        self.userPrefGrpBox.setTitle(_translate("Training", "User Preferences"))
        self.selectLabelColumnLabel.setText(_translate("Training", "Select Target/Label Column"))
        self.HPSeachGrpBox.setTitle(_translate("Training", "HyperParameter Search"))
        self.rationButtonExtensiveSearch.setText(_translate("Training", "Extensive Search"))
        self.radioButtonRandomSearch.setText(_translate("Training", "Random Search"))
        self.enhaceModelGrpBox.setTitle(_translate("Training", "Select Models with"))
        self.radioButtonInterpretability.setText(_translate("Training", "Interpretability"))
        self.radioButtonAccuracy.setText(_translate("Training", "Performance"))
        self.radioButtonSpeed.setText(_translate("Training", "Speed"))
        self.deleteComboBoxLabel.setText(_translate("Training", "Select Features/Culumns to delete from the training "))
        self.runPipelinesBtn.setText(_translate("Training", "Start Training"))
        self.createPipelinesBtn.setText(_translate("Training", "Create Pipelines"))
        self.backBtn.setText(_translate("Training", "Back"))
        self.runPredictionsBtn.setText(_translate("Training", "Make Predictions"))

        # #######################################
        # ########## DATASET FUNCTIONS ##########
        # #######################################

    def uploadDataset(self):
        # Upload the dataset's path
        dialog = QFileDialog()
        filepath = dialog.getOpenFileName(os.getenv('HOME'))
        path = filepath[0]
        if path != "":
            # clear the previous data
            self.comboBoxTargetColumn.clear()
            self.comboBoxDeleteColumns.clear()
            self.textViewFeatureSize.clear()
            self.textViewSampleSize.clear()
            self.datasetPathLineEdit.clear()
            self.datasetPathLineEdit.setText(path)
            self.datasetDetailsBtn.setEnabled(True)
            self.newProjectsPathLineEdit.clear()
            self.openDatasetProfileBtn.setEnabled(False)

    def openDatasetProfiling(self):
        import webbrowser
        main_folder = self.newProjectsPathLineEdit.text()
        path = self.datasetPathLineEdit.text()
        dataset_name_ext = os.path.basename(path)
        dataset_name = os.path.splitext(dataset_name_ext)[0]
        html_file2 = main_folder + "/" "" + dataset_name + "" + "_profile" + "/your_report.html"
        if os.path.isfile(html_file2):
            webbrowser.open(html_file2, new=2)

        # ####################################
        # ##### GET USER PREFERENCES #########
        # ####################################

    def datasetDetails(self):
        sample_size, feature_size, numerical_columns, categorical_columns, missing_values_number, dataset_name, \
            categorical_columns_with_numerical_values, column_list, num_columns_with_cat_values = self.data_analyzer.dataset_details()
        return sample_size, feature_size, numerical_columns, categorical_columns, missing_values_number, dataset_name, \
            categorical_columns_with_numerical_values, column_list, num_columns_with_cat_values

    def saveDeleteComboBox(self):
        sample_size, feature_size, numerical_columns, categorical_columns, missing_values_number, dataset_name, \
            categorical_columns_with_numerical_values, column_list, num_columns_with_cat_values = self.datasetDetails()
        text = self.comboBoxDeleteColumns.currentText()
        new_text = text.split(": ")[1]
        get_index = new_text.split(",")
        labels = []
        column_list.insert(0, "None")
        for i in get_index:
            labels.append(column_list[int(i)])
        for i in labels:
            if "None" in i:
                labels.clear()

    def getColumnsToDelete(self):
        import fnmatch
        # these are the columns in the dataset with uniform unique values
        path = self.datasetPathLineEdit.text()
        dataset_name_ext = os.path.basename(path)
        dataset_name = os.path.splitext(dataset_name_ext)[0]
        uniform_unique_columns = []
        main_folder = self.newProjectsPathLineEdit.text()
        with open("" + main_folder + "/" "" + dataset_name + "" + "_profile" + "/your_report.json") as data_file:
            data = json.load(data_file)
            # get uniform unique columns
            for key, value in data.items():
                if key == "variables":
                    if type(value) == dict:
                        for item, item1 in value.items():
                            if fnmatch.fnmatch(item, "?id" or "?ID" or "?Id"):
                                uniform_unique_columns.append(item)
                            if type(item1) == dict:
                                for element1, element2 in item1.items():
                                    if element1 == "is_unique":
                                        if element2 == bool(True):
                                            uniform_unique_columns.append(item)
                                    if element1 == "type":
                                        if element2 == "Unsupported":
                                            uniform_unique_columns.append(item)
        return uniform_unique_columns

    def getUserPreferences(self):
        sample_size, feature_size, numerical_columns, categorical_columns, missing_values_number, dataset_name, \
            categorical_columns_with_numerical_values, column_list, num_columns_with_cat_values = self.datasetDetails()
        column_list.insert(0, "None")
        # These are the user preferences
        # Check box for Speed user preferences
        interpretability = self.radioButtonInterpretability.isChecked()
        # Radio btn for the Accuracy
        accuracy = self.radioButtonAccuracy.isChecked()
        # Radio btn for interpretability
        speed = self.radioButtonSpeed.isChecked()
        # Radio btn for extensive search
        extensive = self.rationButtonExtensiveSearch.isChecked()
        # Radio btn for random search
        random = self.radioButtonRandomSearch.isChecked()
        # Combo box for the target column
        target_column_type = str(self.comboBoxTargetColumn.currentText())
        # These are the dataset details
        feature_size = self.textViewFeatureSize.text()
        sample_size = self.textViewSampleSize.text()
        # These are the selected columns that user wants to delete
        labels = []
        selected_columns_to_delete = str(self.comboBoxDeleteColumns.currentText())
        if selected_columns_to_delete == "None":
            labels = []
        else:
            new_selected_delete_columns = selected_columns_to_delete.split(":")[1]
            get_index = new_selected_delete_columns.split(",")
            for i in get_index:
                labels.append(column_list[int(i)])
            if "None" in labels:
                labels.clear()
        if (interpretability or accuracy or speed) and (extensive or random):
            return speed, accuracy, interpretability, extensive, random, target_column_type, feature_size, sample_size, \
                labels
        else:
            exception = QMessageBox()
            exception.setWindowTitle("Error")
            exception.setIcon(QMessageBox.Information)
            exception.setText('Please add your pipeline preferences.')
            x = exception.exec_()
            return None

        # #######################################
        # ####### CREATE ONTOLOGY INDIVIDUAL ####
        # #######################################
    @staticmethod
    def load_config():
        # Get the directory of the current script (create.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Path to the configuration file
        config_path = os.path.join(script_dir, '..', 'conf_files', 'conf.json')
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        return config

    def get_file_path(self, config_key):
        # Load the configuration file
        config = self.load_config()
        # Get the directory of the main directory
        main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        # Construct the path to the file specified in the config file
        file_path = os.path.join(main_dir, config[config_key])
        return file_path

    def createIndividual(self):
        path = self.datasetPathLineEdit.text()
        uniform_unique_columns = self.getColumnsToDelete()
        if self.getUserPreferences() is None:
            return
        else:
            speed, accuracy, interpretability, extensive, random, target_column_type, feature_size, sample_size, \
                new_selected_delete_columns = self.getUserPreferences()
            sample_size, feature_size, numerical_columns, categorical_columns, missing_values_number, dataset_name, \
                categorical_columns_with_numerical_values, column_list, num_columns_with_cat_values = self.datasetDetails()
            # Load the ontology (load ntriples or xml to get the rules)
            world = World()
            # file_path = '/OntoAML/main/ontology/version14.rdf'
            ontology_path = self.get_file_path('ontology_path')
            if not ontology_path:
                print("Ontology not found")
                return
            onto = world.get_ontology(ontology_path).load()
            # onto = world.get_ontology("file:///Users/micha/PycharmProjects/MLPipeline/version14.rdf").load()
            # create a new dataset individual
            new_dataset = onto.Dataset(dataset_name + "_Dataset")
            new_dataset.hasName = [dataset_name]
            new_dataset.hasPath = [path]
            new_dataset.hasSampleSize = [int(sample_size)]
            new_dataset.hasFeatureSize = [int(feature_size)]
            # add missing values if there are into the given dataset
            if missing_values_number:
                new_dataset.hasMissingValues = [bool(True)]
            else:
                new_dataset.hasMissingValues = [bool(False)]
            # add if the dataset has categorical or numerical values
            # if categorical_columns:
            ##############################
            ##############################
            ##############################
            if (target_column_type in categorical_columns) and len(categorical_columns) == 1:
                new_dataset.hasCategoricalData = [bool(False)]
            elif (target_column_type in categorical_columns) and len(categorical_columns) > 1:
                new_dataset.hasCategoricalData = [bool(True)]
            elif (target_column_type in numerical_columns) and len(categorical_columns) > 1:
                new_dataset.hasCategoricalData = [bool(True)]
            elif (target_column_type in numerical_columns) and len(categorical_columns) == 1:
                new_dataset.hasCategoricalData = [bool(True)]
            else:
                new_dataset.hasCategoricalData = [bool(False)]
            if numerical_columns:
                new_dataset.hasNumericalData = [bool(True)]
            else:
                new_dataset.hasNumericalData = [bool(False)]
            # add target column name
            new_dataset.hasTargetColumnName = [target_column_type]
            # add columns with unique values
            new_dataset.columnWithUniqueValues = []
            for column in uniform_unique_columns:
                new_dataset.columnWithUniqueValues.append(str(column))
            # add columns to delete by user
            new_dataset.columnsToDelete = []
            for i in new_selected_delete_columns:
                new_dataset.columnsToDelete.append(str(i))
            # create a new user preference individual
            new_pref = onto.UserPreference(dataset_name + "_UserPreferences")
            new_pref_settings = []
            if speed:
                new_pref_settings.append(onto.Speed)
            if accuracy:
                new_pref_settings.append(onto.Accuracy)
            if interpretability:
                new_pref_settings.append(onto.Interpretability)
            if extensive:
                new_pref_settings.append(onto.ExtensiveSearch)
            if random:
                new_pref_settings.append(onto.RandomSearch)
            # check if the selected target column is numerical or categorical
            if target_column_type in categorical_columns:
                new_pref_settings.append(onto.Categorical_Target_Column)
            elif (target_column_type in numerical_columns) and (target_column_type in num_columns_with_cat_values):
                new_pref_settings.append(onto.Categorical_Target_Column)
            elif (target_column_type in numerical_columns) and (target_column_type not in num_columns_with_cat_values):
                new_pref_settings.append(onto.Numerical_Target_Column)
            new_pref.hasUserPreferenceOption = new_pref_settings
            # save the new individuals into a new ntriples ontology
            onto.save(file="reasonerTest6.nt", format="ntriples")
            # self.runReasonerBtn.setEnabled(True)
            # if new_dataset and new_pref:
            #     msg = QMessageBox()
            #     msg.setWindowTitle("Save confirmed")
            #     msg.setText("Your preferences and dataset have been saved!")
            #     msg.setIcon(QMessageBox.Information)
            #     x = msg.exec_()
            return new_dataset, new_pref

        # ##############################################
        # ############## RUN REASONER ##################
        # ########### CREATE THE PIPELINES #############
        # ##############################################
    # Patch the original function

    def runReasoner(self):
        if self.getUserPreferences() is None:
            return
        else:
            self.createIndividual()
            main_folder = self.newProjectsPathLineEdit.text()
            print(main_folder)
            sample_size, feature_size, numerical_columns, categorical_columns, missing_values_number, dataset_name, \
                categorical_columns_with_numerical_values, column_list, num_columns_with_cat_values = self.datasetDetails()
            # world2 = World()
            # list_of_files = glob.glob('C:/Users/micha/PycharmProjects/MLPipeline/*.nt')
            # latest_file = max(list_of_files, key=os.path.getctime)
            # print(latest_file)
            self.runPipelinesBtn.setEnabled(True)
            # call the ontology preferred in n_triples
            # Set the Java heap size to 1GB
            # owlready2.reasoning.sync_reasoner_pellet = self.sync_reasoner_pellet_modified
            world2 = World()
            inferences = world2.get_ontology(r"C:\Users\michalis.g\Desktop\PhD\OntoAML\main\reasonerTest6.nt").load()
            with inferences:
                sync_reasoner_pellet(world2, infer_property_values=True, infer_data_property_values=True)

            # get the inferences of the pipeline
            pipelines = inferences.get_instances_of(inferences.Machine_Learning_Pipeline)

            # get the inferences of the dataset
            dataset = inferences.get_instances_of(inferences.Dataset)[0]

            # get the inferences of the machine learning task class
            ml_task = inferences.get_instances_of(inferences.Machine_Learning_Task)[0]

            # get the Machine Learning problem
            machine_learning_problem = ""

            if ml_task.hasMachineLearningProblem:
                task_problem = str(ml_task.hasMachineLearningProblem[0])
                b = task_problem.split(".")
                machine_learning_problem = b[2]

            # shap explainer

            # target column selected by the user and inferred
            target_column = ""

            # selected columns to delete
            selected_columns_to_delete = []

            # list of the inferred columns with unique values
            uniform_unique_columns = []

            # list of columns that must be deleted before training
            columns_to_delete = []

            # name of the dataset
            dataset_name = ""

            # path of the dataset
            dataset_path = ""

            if dataset.hasName:
                dataset_name = dataset.hasName[0]
            if dataset.hasPath:
                dataset_path = dataset.hasPath[0]
            if dataset.hasTargetColumnName:
                target_column = dataset.hasTargetColumnName[0]
            if dataset.columnsToDelete:
                for column in dataset.columnsToDelete:
                    selected_columns_to_delete.append(str(column))
            if dataset.columnWithUniqueValues:
                for column in dataset.columnWithUniqueValues:
                    uniform_unique_columns.append(str(column))
            columns_to_delete.extend(uniform_unique_columns)
            columns_to_delete.append(target_column)
            columns_to_delete.extend(selected_columns_to_delete)

            # Selects all the inferred pipelines who have at least one modeling algorithm
            inferred_pipelines_with_model = []
            for pipeline in pipelines:
                if pipeline.hasModelingAlgorithm:
                    inferred_pipelines_with_model.append(pipeline)
            print(inferred_pipelines_with_model)
            # For each inferred pipeline who has modeling algorithm create a python based pipeline
            for inferred_pipeline in inferred_pipelines_with_model:
                # list for each inferred algorithm point for the imports
                pipeline_points = []
                # list of each inferred pre_processing algorithm point for the preprocess pipeline
                pre_proc_pipeline_points = []
                # list of each inferred feature selection and model algorithm point for the main pipeline
                fs_pipeline_points = []
                # list of inferred dimensionality reduction points
                dim_redu_pipeline_points = []
                # list of each inferred machine learning model
                model_pipeline_points = []
                # shap_explainer_type
                shap_explainer_type = []
                # hyperparameter_tuning_type
                hyper_parameter_tuning_type = []
                # name of the inferred model algorithm point to use it in the interpretability
                model_point = ""
                # name of the inferred feature_selection point to use it in the int interpretability
                fs_point = ""
                if inferred_pipeline.hasDataPreProcessingAlgorithm:
                    for data_preprocessing_algorithms in inferred_pipeline.hasDataPreProcessingAlgorithm:
                        pipeline_points.append(str(data_preprocessing_algorithms.pointsTo[0]))
                        pre_proc_pipeline_points.append(str(data_preprocessing_algorithms.pointsTo[0]))
                if inferred_pipeline.hasDataProcessingAlgorithm:
                    for data_processing_algorithms in inferred_pipeline.hasDataProcessingAlgorithm:
                        pipeline_points.append(str(data_processing_algorithms.pointsTo[0]))
                        fs_pipeline_points.append(data_processing_algorithms.pointsTo[0])
                        fs_point = data_processing_algorithms.pointsTo[0]
                if inferred_pipeline.hasDimensionalityReductionAlgorithm:
                    for dimensionality_reduction_algorithms in inferred_pipeline.hasDimensionalityReductionAlgorithm:
                        pipeline_points.append(str(dimensionality_reduction_algorithms.pointsTo[0]))
                        fs_pipeline_points.append(str(dimensionality_reduction_algorithms.pointsTo[0]))
                        dim_redu_pipeline_points.append(str(dimensionality_reduction_algorithms.pointsTo[0]))
                if inferred_pipeline.hasModelingAlgorithm:
                    for model_algorithms in inferred_pipeline.hasModelingAlgorithm:
                        pipeline_points.append(str(model_algorithms.pointsTo[0]))
                        model_pipeline_points.append(str(model_algorithms.pointsTo[0]))
                        model_point = model_algorithms.pointsTo[0]
                if inferred_pipeline.hasHyperParameterTuningAlgorithm:
                    for hyper_parameter_tuning_algorithm in inferred_pipeline.hasHyperParameterTuningAlgorithm:
                        hyper_parameter_tuning_type.append(str(hyper_parameter_tuning_algorithm.pointsTo[0]))

                if inferred_pipeline.hasExplainabilityMethod:
                    for explainable_method in inferred_pipeline.hasExplainabilityMethod:
                        shap_explainer_type.append(str(explainable_method.pointsTo[0]))

                # set the imputation algorithm first
                if "sklearn.impute.SimpleImputer" in pre_proc_pipeline_points:
                    imputer = pre_proc_pipeline_points.index("sklearn.impute.SimpleImputer")
                    pre_proc_pipeline_points[0], pre_proc_pipeline_points[imputer] = \
                        pre_proc_pipeline_points[imputer], pre_proc_pipeline_points[0]

                # # set the one hot encoder second
                # if "sklearn.preprocessing.OneHotEncoder" in pre_proc_pipeline_points:
                #     one_hot_en = pre_proc_pipeline_points.index("sklearn.preprocessing.OneHotEncoder")
                #     pre_proc_pipeline_points[1], pre_proc_pipeline_points[one_hot_en] = \
                #         pre_proc_pipeline_points[one_hot_en], pre_proc_pipeline_points[1]

                # set the one hot encoder second
                if "sklearn.preprocessing.StandardScaler" in pre_proc_pipeline_points:
                    scaler = pre_proc_pipeline_points.index("sklearn.preprocessing.StandardScaler")
                    pre_proc_pipeline_points[-1], pre_proc_pipeline_points[scaler] = \
                        pre_proc_pipeline_points[scaler], pre_proc_pipeline_points[-1]

                selected_hp_tuning = hyper_parameter_tuning_type[0]
                # creates a directory with the pipeline
                for model in model_pipeline_points:
                    model_name_only = str(model.split(".")[2])
                    if os.path.exists(os.path.join("" + main_folder + "/" + "" + model_name_only + ""
                                                   + "_Pipeline")):
                        continue
                    else:
                        os.mkdir(os.path.join("" + main_folder + "/" + "" + model_name_only + ""
                                              + "_Pipeline"))
                    f = open(os.path.join("" + main_folder + "/" + "" + model_name_only + ""
                                          + "_Pipeline", "" + model_name_only + "" + "_pipeline.py"), "w+")
                    # write the main imports of the file
                    f.write(PythonPipelineEnums.pipeline_imports.value)
                    f.write("\n")  # change line
                    f.write("# Imports from the selected algorithms\n")
                    # call the imports of each inferred primitive
                    for point in pre_proc_pipeline_points:
                        for key, value in conf.model_import_conf.items():
                            if key == point:
                                if type(value) == dict:
                                    for element1, element2 in value.items():
                                        f.write(str(element2) + "\n")
                                else:
                                    f.write(str(value) + "\n")
                    for point in fs_pipeline_points:
                        for key, value in conf.model_import_conf.items():
                            if key == point:
                                if type(value) == dict:
                                    for element1, element2 in value.items():
                                        f.write(str(element2) + "\n")
                                else:
                                    f.write(str(value) + "\n")
                    for point in model_pipeline_points:
                        if point == model:
                            for key, value in conf.model_import_conf.items():
                                if key == point:
                                    if type(value) == dict:
                                        for element1, element2 in value.items():
                                            f.write(str(element2) + "\n")
                                    else:
                                        f.write(str(value) + "\n")
                    f.write("\n")  # change line
                    # set the chosen dataset
                    f.write("missing_values_formats = ['n/a', 'na', '--', '?', ' ', 'NA', 'N/A']\n")
                    f.write('separation = "[,]+|;|:"\n')
                    f.write("df = pd.read_csv(\'" + dataset_path + "\', na_values=missing_values_formats, sep=None, "
                                                                   "engine='python', encoding='UTF-8')\n")
                    f.write("df = df.dropna(subset=[\'" + target_column + "\'""])" + "\n")
                    f.write("X = df.drop([")
                    for i in columns_to_delete:
                        f.write("\'" + i + "\', ")
                    f.write("], axis=1)" + "\n")
                    f.write("y = df[\'" + target_column + "\'""]" + "\n")
                    f.write("num_features = [")
                    for i in numerical_columns:
                        if i in columns_to_delete:
                            continue
                        elif i == target_column:
                            continue
                        else:
                            f.write("\'" + i + "\', ")
                    f.write("]\n")
                    f.write("cat_features = [")
                    categorical_features1 = []
                    for i in categorical_columns:
                        if i in columns_to_delete:
                            continue
                        elif i == target_column:
                            continue
                        else:
                            categorical_features1.append(i)
                            f.write("\'" + i + "\', ")
                    f.write("]\n")
                    categorical_features_true = []
                    if len(categorical_features1) == 0:
                        categorical_features_true.append(0)
                    else:
                        categorical_features_true.append(1)
                    # write categorical features with numerical values (this helps interpretability(lime))
                    f.write("categorical_features_with_numerical_values = [")
                    for i in categorical_columns_with_numerical_values:
                        if i in columns_to_delete:
                            continue
                        elif i == target_column:
                            continue
                        else:
                            f.write("\'" + i + "\', ")
                    f.write("]\n")
                    # convert the columns as numerical or categorical
                    f.write("df[num_features] = df[num_features].apply(pd.to_numeric, errors='coerce')\n")
                    f.write("df[cat_features] = df[cat_features].astype(object)\n")
                    f.write("\n")
                    f.write("# call the names of the selected machine learning algorithms\n")
                    # call the names of the primitives
                    for point in pre_proc_pipeline_points:
                        for key, value in conf.model_names_for_pipeline_dict.items():
                            if key == point and key == "sklearn.impute.SimpleImputer":
                                f.write(value + " = " + value + "\n")
                                continue
                            elif key == point and key == "sklearn.preprocessing.OneHotEncoder":
                                f.write(value + " = " + value + "\n")
                                continue
                            elif key == point:
                                # f.write(value + " = " + value + "\n")
                                f.write(value + " = " + value + "()" + "\n")
                    for point in fs_pipeline_points:
                        for key, value in conf.model_names_for_pipeline_dict.items():
                            if key == point and key == 'sklearn.feature_selection.SelectFromModel':
                                continue
                            elif key == point and key == 'sklearn.feature_selection.RFE':
                                continue
                            elif key == point and key == 'sklearn.decomposition.PCA':
                                continue
                            elif key == point and key == 'sklearn.feature_selection.SelectKBest.chi2':
                                continue
                            elif key == point and key == 'sklearn.feature_selection.RFE1':
                                continue
                            elif key == point and key == 'sklearn.feature_selection.SelectFromModel_L1':
                                continue
                            elif key == point and key == 'sklearn.feature_selection.SelectKBest_kr':
                                continue
                            elif key == point and key == 'sklearn.feature_selection.ElasticNet':
                                continue
                            elif key == point:
                                # if type(value) == dict:
                                #     for element1, element2 in value.items():
                                #         f.write(str(element1) + " = " + str(element2) + "\n")
                                # else:
                                f.write(value + " = " + value + "\n")
                                # f.write(value + " = " + value + "()" + "\n")
                    for key, value in conf.model_names_for_pipeline_dict.items():
                        if key == model:
                            f.write(value + " = " + value + "\n")
                            # f.write(value + " = " + value + "()" + "\n")
                    # f.write("\n\n")  # change line
                    # create the column transformer for numerical data
                    f.write(PythonPipelineEnums.pipeline_comments.value)
                    f.write("numeric_transformer = Pipeline(steps=[\n")
                    if len(pre_proc_pipeline_points) == 0:
                        f.write("   ('pass', 'passthrough')" + "\n")
                        f.write("])\n\n")
                    else:
                        for point in pre_proc_pipeline_points:
                            # create the pipeline
                            for key, value in conf.model_names_for_pipeline_dict.items():
                                if key == point and key == "sklearn.impute.SimpleImputer":
                                    f.write("   (\'" + value + "\'" + ", " + value + "(strategy='median'))," + "\n")
                                    continue
                                elif (key == point) and (key == "sklearn.preprocessing.OneHotEncoder") and \
                                        len(pre_proc_pipeline_points) == 1:
                                    f.write("   ('pass', 'passthrough')" + "\n")
                                    continue
                                elif key == point and key == "sklearn.preprocessing.OneHotEncoder":
                                    pass
                                elif key == point:
                                    f.write("   (\'" + value + "\'" + ", " + value + ")," + "\n")
                        f.write("])\n\n")
                    # create the column transformer for categorical data
                    f.write("categorical_transformer = Pipeline(steps=[\n")
                    if categorical_features_true[0] == 0:
                        print("true")
                        f.write("   ('pass', 'passthrough')" + "\n")
                        f.write("])\n\n")
                    else:
                        print("false")
                        for point in pre_proc_pipeline_points:
                            # create the pipeline
                            for key, value in conf.model_names_for_pipeline_dict.items():
                                if key == point and key == "sklearn.impute.SimpleImputer":
                                    f.write(
                                        "   (\'" + value + "\'" + ", " + value + "(strategy='most_frequent'))," + "\n")
                                    continue
                                # elif (key == point) and (key == "sklearn.preprocessing.OneHotEncoder") and
                                #     len(ca)
                                elif key == point and key == "sklearn.preprocessing.OneHotEncoder":
                                    f.write(
                                        "   (\'" + value + "\'" + ", " + value + "(sparse=False, handle_unknown='ignore')),"
                                        + "\n")
                                    continue
                                elif key == point:
                                    f.write("   (\'" + value + "\'" + ", " + value + ")," + "\n")
                        f.write("])\n\n")

                    if (len(numerical_columns) != 0) and (len(categorical_columns) == 0):
                        # create preprocessor from numeric_transformer with no categorical transformer since there are
                        # no categorical features
                        f.write(PythonPipelineEnums.no_categorical_features_preprocessor.value)
                    elif (len(numerical_columns) == 0) and (len(categorical_columns) != 0):
                        # create preprocessor categorical_transformer without numerical transformer since there are no
                        # numerical features
                        f.write(PythonPipelineEnums.numerical_features_preprocessor.value)
                    elif (len(numerical_columns) != 0) and (len(categorical_columns) != 0):
                        # create preprocessor from numeric_transformer and categorical_transformer
                        f.write(PythonPipelineEnums.numerical_and_cateorical_features_preprocessor.value)
                    # create the pipeline #############################################
                    f.write("pipeline = Pipeline(steps=[\n")
                    f.write("   ('preprocessor', preprocessor),\n")
                    for point in fs_pipeline_points:
                        # create the pipeline
                        for key, value in conf.model_names_for_pipeline_dict.items():
                            if key == point and type(value) == dict:
                                for element1, element2 in value.items():
                                    f.write("   (\'" + element1 + "\'" + ", " + element2 + ")," + "\n")
                                    continue
                            elif key == point:
                                f.write("   (\'" + value + "\'" + ", " + value + "()" + ")," + "\n")
                    for key, value in conf.model_names_for_pipeline_dict.items():
                        if key == model:
                            f.write("   (\'" + value + "\'" + ", " + value + "()" + ")," + "\n")
                    f.write("])\n\n")

                    # list the hyperparameters of each primitive of the pipeline
                    f.write("# This is the hyper-parameter search space based on the above algorithms\n")
                    f.write("param_grid = { \n")
                    grid_params = {}
                    for point in fs_pipeline_points:
                        for key, value in conf.algorithms_config_dict.items():
                            if key == point:
                                for element1, element2 in value.items():
                                    if type(element2) == dict:
                                        for element3, element4 in element2.items():
                                            if type(element4) == dict:
                                                for element5, element6 in element4.items():
                                                    if type(element6) == list:
                                                        f.write(
                                                            "{}".format(
                                                                str("   \'" + element5 + "\'") + ": " + str(element6) +
                                                                ",\n"))
                                                        grid_params.update({element5: element6})
                                                    if type(element6) == np.ndarray:
                                                        f.write("{}".format(
                                                            str("   \'" + element5 + "\'") + ": [" + str(
                                                                ', '.join(map(str, element6)))
                                                            + "],\n"))
                                                        grid_params.update({element5: element6})
                                                    if type(element6) == range:
                                                        f.write("{}".format(
                                                            str("   \'" + element5 + "\'") + ": [" + str(
                                                                ', '.join(map(str, element6)))
                                                            + "],\n"))
                                                        grid_params.update({element5: element6})
                                            if type(element4) == list:
                                                f.write(
                                                    "{}".format(
                                                        str("   \'" + element3 + "\'") + ": " + str(element4) + ",\n"))
                                                grid_params.update({element3: element4})
                                            if type(element4) == np.ndarray:
                                                f.write("{}".format(
                                                    str("   \'" + element3 + "\'") + ": [" + str(
                                                        ', '.join(map(str, element4))) +
                                                    "],\n"))
                                                grid_params.update({element3: element4})
                                            if type(element4) == range:
                                                f.write("{}".format(
                                                    str("   \'" + element3 + "\'") + ": [" + str(
                                                        ', '.join(map(str, element4))) +
                                                    "],\n"))
                                                grid_params.update({element3: element4})
                                    if type(element2) == list:
                                        f.write(
                                            "{}".format(str("   \'" + element1 + "\'") + ": " + str(element2) + ",\n"))
                                        grid_params.update({element1: element2})
                                    if type(element2) == np.ndarray:
                                        f.write("{}".format(
                                            str("   \'" + element1 + "\'") + ": [" + str(', '.join(map(str, element2)))
                                            + "],\n"))
                                        grid_params.update({element1: element2})
                                    if type(element2) == range:
                                        f.write("{}".format(
                                            str("   \'" + element1 + "\'") + ": [" + str(', '.join(map(str, element2)))
                                            + "],\n"))
                                        grid_params.update({element1: element2})
                    # selected model hyperparamerers
                    for point in model_pipeline_points:
                        if point == model:
                            for key, value in conf.algorithms_config_dict.items():
                                if key == point:
                                    for element1, element2 in value.items():
                                        if type(element2) == dict:
                                            for element3, element4 in element2.items():
                                                if type(element4) == dict:
                                                    for element5, element6 in element4.items():
                                                        if type(element6) == list:
                                                            f.write("{}".format(
                                                                str("   \'" + element5 + "\'") + ": " + str(element6)
                                                                + ",\n"))
                                                            grid_params.update({element5: element6})
                                                        if type(element6) == np.ndarray:
                                                            f.write("{}".format(
                                                                str("   \'" + element5 + "\'") + ": ["
                                                                + str(', '.join(map(str, element6))) + "],\n"))
                                                            grid_params.update({element5: element6})
                                                        if type(element6) == range:
                                                            f.write("{}".format(
                                                                str("   \'" + element5 + "\'") + ": ["
                                                                + str(', '.join(map(str, element6))) + "],\n"))
                                                            grid_params.update({element5: element6})
                                                if type(element4) == list:
                                                    f.write("{}".format(
                                                        str("   \'" + element3 + "\'") + ": " + str(element4) + ",\n"))
                                                    grid_params.update({element3: element4})
                                                if type(element4) == np.ndarray:
                                                    f.write("{}".format(
                                                        str("   \'" + element3 + "\'") + ": [" + str(
                                                            ', '.join(map(str, element4))) +
                                                        "],\n"))
                                                    grid_params.update({element3: element4})
                                                if type(element4) == range:
                                                    f.write("{}".format(
                                                        str("   \'" + element3 + "\'") + ": [" + str(
                                                            ', '.join(map(str, element4))) +
                                                        "],\n"))
                                                    grid_params.update({element3: element4})
                                        if type(element2) == list:
                                            f.write(
                                                "{}".format(
                                                    str("   \'" + element1 + "\'") + ": " + str(element2) + ",\n"))
                                            grid_params.update({element1: element2})
                                        if type(element2) == np.ndarray:
                                            f.write("{}".format(str("   \'" + element1 + "\'") + ": ["
                                                                + str(', '.join(map(str, element2))) + "],\n"))
                                            grid_params.update({element1: element2})
                                        if type(element2) == range:
                                            f.write("{}".format(str("   \'" + element1 + "\'")
                                                                + ": [" + str(', '.join(map(str, element2))) + "],\n"))
                                            grid_params.update({element1: element2})
                    f.write(" }\n")
                    # print the data_set split and use the label encoder to the y
                    if machine_learning_problem == "Classification":
                        f.write(PythonPipelineEnums.classification_ml_problem.value)
                    elif machine_learning_problem == "Regression":
                        f.write(PythonPipelineEnums.data_split.value)
                    # take the name of the model
                    model_name = ""
                    model_name2 = ""
                    model_name3 = ""
                    model_name4 = ""
                    if model:
                        for key, value in conf.model_names_for_pipeline_dict.items():
                            if model == key:
                                model_name = repr(value)
                                # variable to save the pipeline by its model name
                                model_name2 = repr(value + "_trained_pipeline.joblib")
                                model_name3 = repr(value + "_trained_pipeline_scores.txt")
                                model_name4 = repr(value + "_trained_pipeline_scores.html")
                    # #####################################
                    # #####################################
                    if machine_learning_problem == "Classification":
                        if selected_hp_tuning == "GridSearchCV" and len(grid_params) > 8:
                            # use RandomSearchCV if the hyper_parameter are greater than 9 and create metrics file
                            # for the pipeline
                            f.write("""
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, random_state=0, scoring="accuracy", 
                                   n_jobs=2, cv=5)
random_search.fit(X_train, Y_train)
file_dir = os.path.dirname(__file__)

# Calculate the mean and standard deviation of the scores
mean_score = np.mean(random_search.cv_results_['mean_test_score'])
std_score = np.std(random_search.cv_results_['mean_test_score'])

# Get the minimum and maximum scores
min_score = np.min(random_search.cv_results_['mean_test_score'])
max_score = np.max(random_search.cv_results_['mean_test_score'])


pipeline.set_params(**random_search.best_params_)
pipeline.get_params(""" + model_name + """)
pipeline.fit(X_train, Y_train)

# Evaluate the model on the training set
train_acc = accuracy_score(Y_train, pipeline.predict(X_train))
val_acc = accuracy_score(Y_test, pipeline.predict(X_test))
overfit_estimation = train_acc - val_acc

# Generate a learning curve to estimate the overfit
train_sizes, train_scores, test_scores = learning_curve(pipeline, X, Y_all, cv=5, n_jobs=-1,
                                                        train_sizes=np.linspace(0.1, 1.0, 10))
# Calculate the training and testing accuracy with learning curve
train_acc_learning_curve = np.mean(train_scores, axis=1)
val_acc_learning_curve = np.mean(test_scores, axis=1)

# Calculate the overfit gap
overfit_gap_learning_curve = train_acc_learning_curve - val_acc_learning_curve

# Get the last overfit value gap
last_overfit_gap = overfit_gap_learning_curve[-1]

# Plot the learning curve
plt.grid()
plt.plot(train_sizes, train_acc_learning_curve, label='Training accuracy')
plt.plot(train_sizes, val_acc_learning_curve, label='Testing accuracy')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.xlabel('Training size')
plt.ylabel('Accuracy')
# plt.ylim(0, 1.00)
ax = plt.gca()
ax.set_yticks(np.arange(0, 1.10, 0.05))
plt.title('Learning Curve')

# Plot the overfit gap
plt.twinx()
plt.plot(train_sizes, overfit_gap_learning_curve, color='r', label='Overfit gap')
plt.legend(loc='lower left')
plt.ylabel('Overfit gap')

# Save the overfit plot
plot_image = os.path.join(file_dir, 'overfitting_plot.png')
plt.savefig(plot_image, dpi=250, format='png', bbox_inches='tight')

# Save Pipeline diagram in HTML
set_config(display="diagram")
pipeline_diagram = estimator_html_repr(pipeline)
pipeline_diagram_file = open(os.path.join(file_dir, 'pipeline_diagram.html'), "w", encoding='utf-8')
pipeline_diagram_file.write(pipeline_diagram)

# ###################################
# ######### METRICS #################
# ###################################

# Make classification report image
def show_values(pc, fmt="%.2f", **kw):
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20,
            correct_orientation=False, cmap='RdBu'):

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))
        

def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    lines = classification_report.split('\\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 4)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height,
            correct_orientation, cmap=cmap)


y_true = Y_test
y_pred = pipeline.predict(X_test)
if (target_labels == [1, 0]) or (target_labels == [0, 1]) or (target_labels == [1, 2]) or (target_labels == [2, 1]) or (target_labels == [-1, 1]) or (target_labels == [1, -1]):
    report = metrics.classification_report(y_true, y_pred)
else:
    report = metrics.classification_report(y_true, y_pred, target_names=target_labels)
print(report)

# Plot classification report and save in figure
plot_classification_report(report)
classification_report_plot_name = os.path.join(file_dir, 'test_plot_classif_report.png')
plt.savefig(classification_report_plot_name, dpi=200, format='png', bbox_inches='tight')
# plt.savefig(file_dir + 'test_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')
plt.close()

# Plot Roc curve and save it in figure
metrics.RocCurveDisplay.from_predictions(y_true, y_pred)
roc_curve_plot_name = os.path.join(file_dir, "plot_roc_curve.png")
plt.savefig(roc_curve_plot_name, dpi=200, format='png', bbox_inches='tight')
# plt.savefig(file_dir + 'plot_roc_curve.png', dpi=200, format='png', bbox_inches='tight')
plt.close()

classification_report_plot = os.path.abspath('test_plot_classif_report.png')
roc_curve_plot = os.path.abspath('plot_roc_curve.png')
overfitting_plot = os.path.abspath('overfitting_plot.png')


def print_report(random_search):
    # TXT report
    filename_scores = open(os.path.join(file_dir, """ + model_name3 + """), "w", encoding='utf-8')
    y_true = Y_test
    y_pred = pipeline.predict(X_test)
    # report = metrics.classification_report(y_true, y_pred, target_names=target_labels)
    filename_scores.write("Problem: Classification" + "\\n")
    filename_scores.write("model's name: " """ + model_name + """ + "\\n")
    filename_scores.write(report + "\\n")
    filename_scores.write("Auc score: {:0.3f}".format(metrics.roc_auc_score(Y_test, y_pred)) + "\\n")
    filename_scores.write("Overfit estimation with different {:?f}" + "\\n")
    filename_scores.write("{:0.1f}".format(overfit_estimation) + "\\n")
    filename_scores.write("{:0.2f}".format(overfit_estimation) + "\\n")
    filename_scores.write("{:0.3f}".format(overfit_estimation) + "\\n")
    filename_scores.write("{:0.4f}".format(overfit_estimation) + "\\n")
    filename_scores.write("Learning curve with different {:?f} for the overall mean value"+ "\\n")
    filename_scores.write("{:0.1f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("{:0.2f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("{:0.3f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("{:0.4f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("Learning curve with different {:?f} for the last value" + "\\n")
    filename_scores.write("{:0.1f}".format(last_overfit_gap) + "\\n")
    filename_scores.write("{:0.2f}".format(last_overfit_gap) + "\\n")
    filename_scores.write("{:0.3f}".format(last_overfit_gap) + "\\n")
    filename_scores.write("{:0.4f}".format(last_overfit_gap) + "\\n")
    filename_scores.close()
    # HTML report
    report_dict = metrics.classification_report(y_true, y_pred, target_names=target_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_html = report_df.to_html()
    if std_score < 0.1 * mean_score:
        score_consistensy = "The scores are relatively consistent."
    else:
        score_consistensy = "The scores are relatively inconsistent."
    if train_acc > val_acc:
        overfit = "The model may be overfitting."
    else:
        overfit = "The model does not seem to be overfitting."
    custom_dict_report_html = {
        "Problem": ["Classification"],
        "Target Column Name": [" """ + target_column + """ "],
        "Model's Name": [""" + model_name + """],
        "Accuracy Score": ["{:0.5f}".format(metrics.accuracy_score(y_true, y_pred))],
        "Roc Auc curve": ["{:0.3f}".format(metrics.roc_auc_score(Y_test, y_pred))],
        "Mean accuracy score of each tested hyperparameter combination": ["{:0.3f}".format(mean_score)],
        "Range of all accuracy scores of each tested hyperparameter combination":
            ["{:.3f} - {:.3f}".format(min_score, max_score)],
        "Standard Deviation of scores": ["{:0.3f}".format(std_score)],
        "Standard Deviation < 0.1 * Mean Accuracy scores": ["" + score_consistensy + ""]}
    df_custom_report_html = pd.DataFrame(custom_dict_report_html).transpose()
    custom_report_html = df_custom_report_html.to_html(header=False)
    overfit_report = {
        "Overfit Report": ["The Report is based only on Accuracy"],
        "Train set accuracy score of best pipeline": ["{:0.4f}".format(train_acc)],
        "Test set accuracy score of best pipeline": ["{:0.4f}".format(val_acc)],
        "Overfit estimation score of the best pipeline": ["{:0.4f}".format(overfit_estimation)],
        "Learning Curve scores report": ["The Learning Curve is based on Accuracy"],
        "Train set accuracy score of learning curve's last value": ["{:0.2f}".format(train_acc_learning_curve[-1])],
        "Test set accuracy score of learning curve's last value": ["{:0.2f}".format(val_acc_learning_curve[-1])],
        "Overfit gap of learning curve's last value": ["{:0.2f}".format(last_overfit_gap)]}
    df_overfit_report_html = pd.DataFrame(overfit_report).transpose()
    custom_overfit_report = df_overfit_report_html.to_html(header=False)
    filename_scores_html = open(os.path.join(file_dir, """ + model_name4 + """), "w", encoding='utf-8')
    filename_scores_html.write(custom_report_html)
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Classification Report:" + "<br>")
    filename_scores_html.write(report_html + "<br>")
    filename_scores_html.write('<img src = "' + classification_report_plot + '" alt ="cfg"><br>')
    filename_scores_html.write("<font size= "'6'"><b> " + "Roc Auc curve figure:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + roc_curve_plot + '" alt ="cfg"><br>')
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Overfit Report:" + "<br>")
    filename_scores_html.write(custom_overfit_report + "<br>")
    filename_scores_html.write("<font size= "'6'"><b> " + "Learning Curve - Overfitting or Underfitting:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + overfitting_plot + '" alt ="cfg">')
    filename_scores_html.close()
print_report(pipeline)
filename = os.path.join(file_dir, """ + model_name2 + """)
joblib.dump(pipeline, filename=filename)\n\n""")
                        elif selected_hp_tuning == "GridSearchCV" and len(grid_params) <= 8:
                            # else use GridSearchCv and create metrics file for the pipeline
                            f.write("""
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=5, cv=5, scoring="accuracy", n_jobs=2)
grid_search.fit(X_train, Y_train)


# Calculate the mean and standard deviation of the scores
mean_score = np.mean(grid_search.cv_results_['mean_test_score'])
std_score = np.std(grid_search.cv_results_['mean_test_score'])

# Get the minimum and maximum scores
min_score = np.min(grid_search.cv_results_['mean_test_score'])
max_score = np.max(grid_search.cv_results_['mean_test_score'])

file_dir = os.path.dirname(__file__)
pipeline.set_params(**grid_search.best_params_)
pipeline.get_params(""" + model_name + """)
pipeline.fit(X_train, Y_train)

# Evaluate the model on the training set
train_acc = accuracy_score(Y_train, pipeline.predict(X_train))
val_acc = accuracy_score(Y_test, pipeline.predict(X_test))
overfit_estimation = train_acc - val_acc

# Generate a learning curve to estimate the overfit
train_sizes, train_scores, test_scores = learning_curve(pipeline, X, Y_all, cv=5, scoring="accuracy", n_jobs=-1,
                                                        train_sizes=np.linspace(0.1, 1.0, 10))
# Calculate the training and testing accuracy with learning curve
train_acc_learning_curve = np.mean(train_scores, axis=1)
val_acc_learning_curve = np.mean(test_scores, axis=1)

# Calculate the overfit gap
overfit_gap_learning_curve = train_acc_learning_curve - val_acc_learning_curve

# Get the last value
last_overfit_gap = overfit_gap_learning_curve[-1]

# Plot the learning curve
plt.grid()
plt.plot(train_sizes, train_acc_learning_curve, label='Training accuracy')
plt.plot(train_sizes, val_acc_learning_curve, label='Testing accuracy')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.xlabel('Training size')
plt.ylabel('Accuracy')
# plt.ylim(0, 1.00)
ax = plt.gca()
ax.set_yticks(np.arange(0, 1.10, 0.05))
plt.title('Learning Curve')

# Plot the overfit gap
plt.twinx()
plt.plot(train_sizes, overfit_gap_learning_curve, color='r', label='Overfit gap')
plt.legend(loc='lower left')
plt.ylabel('Overfit gap')

# Save the plot
plot_image = os.path.join(file_dir, 'overfitting_plot.png')
plt.savefig(plot_image, dpi=250, format='png', bbox_inches='tight')

# Save Pipeline diagram in HTML
set_config(display="diagram")
pipeline_diagram = estimator_html_repr(pipeline)
pipeline_diagram_file = open(os.path.join(file_dir, 'pipeline_diagram.html'), "w", encoding='utf-8')
pipeline_diagram_file.write(pipeline_diagram)

# ###################################
# ######### METRICS #################
# ###################################

# Make classification report image
def show_values(pc, fmt="%.2f", **kw):
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)
       
def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, 
            correct_orientation=False, cmap='RdBu'):

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))
        

def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    lines = classification_report.split('\\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 4)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height,
            correct_orientation, cmap=cmap)


y_true = Y_test
y_pred = pipeline.predict(X_test)
if (target_labels == [1, 0]) or (target_labels == [0, 1]) or (target_labels == [1, 2]) or (target_labels == [2, 1]) or (target_labels == [-1, 1]) or (target_labels == [1, -1]):
    report = metrics.classification_report(y_true, y_pred)
else:
    report = metrics.classification_report(y_true, y_pred, target_names=target_labels)
print(report)

# Plot classification report and save in figure
plot_classification_report(report)
classification_report_plot_name = os.path.join(file_dir, 'test_plot_classif_report.png')
plt.savefig(classification_report_plot_name, dpi=200, format='png', bbox_inches='tight')
# plt.savefig(file_dir + 'test_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')
plt.close()

# Plot Roc curve and save it in figure
metrics.RocCurveDisplay.from_predictions(y_true, y_pred)
roc_curve_plot_name = os.path.join(file_dir, "plot_roc_curve.png")
plt.savefig(roc_curve_plot_name, dpi=200, format='png', bbox_inches='tight')
# plt.savefig(file_dir + 'plot_roc_curve.png', dpi=200, format='png', bbox_inches='tight')
plt.close()

classification_report_plot = os.path.abspath('test_plot_classif_report.png')
roc_curve_plot = os.path.abspath('plot_roc_curve.png')
overfitting_plot = os.path.abspath('overfitting_plot.png')


def print_report(grid_search):
    # TXT report
    filename_scores = open(os.path.join(file_dir, """ + model_name3 + """), "w", encoding='utf-8')
    y_true = Y_test
    y_pred = pipeline.predict(X_test)
    # report = metrics.classification_report(y_true, y_pred, target_names=target_labels)
    filename_scores.write("Problem: Classification" + "\\n")
    filename_scores.write("model's name: " """ + model_name + """ + "\\n")
    filename_scores.write(report + "\\n")
    filename_scores.write("Accuracy score: {:0.5f}".format(metrics.accuracy_score(y_true, y_pred)))
    filename_scores.write("\\n")
    filename_scores.write("F1 score: {:0.3f}".format(metrics.f1_score(y_true, y_pred)))
    filename_scores.write("\\n")
    filename_scores.write("Auc score: {:0.3f}".format(metrics.roc_auc_score(Y_test, y_pred)) + "\\n")
    filename_scores.write("Overfit estimation with different {:?f}" + "\\n")
    filename_scores.write("{:0.1f}".format(overfit_estimation) + "\\n")
    filename_scores.write("{:0.2f}".format(overfit_estimation) + "\\n")
    filename_scores.write("{:0.3f}".format(overfit_estimation) + "\\n")
    filename_scores.write("{:0.4f}".format(overfit_estimation) + "\\n")
    filename_scores.write("Learning curve with different {:?f} for the overall mean value"+ "\\n")
    filename_scores.write("{:0.1f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("{:0.2f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("{:0.3f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("{:0.4f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("Learning curve with different {:?f} for the last value" + "\\n")
    filename_scores.write("{:0.1f}".format(last_overfit_gap) + "\\n")
    filename_scores.write("{:0.2f}".format(last_overfit_gap) + "\\n")
    filename_scores.write("{:0.3f}".format(last_overfit_gap) + "\\n")
    filename_scores.write("{:0.4f}".format(last_overfit_gap) + "\\n")
    filename_scores.close()  
    # HTML report
    report_dict = metrics.classification_report(y_true, y_pred, target_names=target_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_html = report_df.to_html()
    if std_score < 0.1 * mean_score:
        score_consistensy = "The scores are relatively consistent."
    else:
        score_consistensy = "The scores are relatively inconsistent."
    if train_acc > val_acc:
        overfit = "The model may be overfitting."
    else:
        overfit = "The model does not seem to be overfitting."
    custom_dict_report_html = {
        "Problem": ["Classification"],
        "Target Column Name": [" """ + target_column + """ "],
        "Model's Name": [""" + model_name + """],
        "Accuracy Score": ["{:0.5f}".format(metrics.accuracy_score(y_true, y_pred))],
        "Roc Auc curve": ["{:0.3f}".format(metrics.roc_auc_score(Y_test, y_pred))],
        "Mean accuracy score of each tested hyperparameter combination": ["{:0.3f}".format(mean_score)],
        "Range of all accuracy scores of each tested hyperparameter combination":
            ["{:.3f} - {:.3f}".format(min_score, max_score)],
        "Standard Deviation of scores": ["{:0.3f}".format(std_score)],
        "Standard Deviation < 0.1 * Mean Accuracy scores": ["" + score_consistensy + ""]}
    df_custom_report_html = pd.DataFrame(custom_dict_report_html).transpose()
    custom_report_html = df_custom_report_html.to_html(header=False)
    overfit_report = {
        "Overfit Report": ["The Report is based only on Accuracy"],
        "Train set accuracy score of best pipeline": ["{:0.4f}".format(train_acc)],
        "Test set accuracy score of best pipeline": ["{:0.4f}".format(val_acc)],
        "Overfit estimation score of the best pipeline": ["{:0.4f}".format(overfit_estimation)],
        "Learning Curve scores report": ["The Learning Curve is based on Accuracy"],
        "Train set accuracy score of learning curve's last value": ["{:0.2f}".format(train_acc_learning_curve[-1])],
        "Test set accuracy score of learning curve's last value": ["{:0.2f}".format(val_acc_learning_curve[-1])],
        "Overfit gap of learning curve's last value": ["{:0.2f}".format(last_overfit_gap)]}
    df_overfit_report_html = pd.DataFrame(overfit_report).transpose()
    custom_overfit_report = df_overfit_report_html.to_html(header=False)
    filename_scores_html = open(os.path.join(file_dir, """ + model_name4 + """), "w", encoding='utf-8')
    filename_scores_html.write(custom_report_html)
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Classification Report:" + "<br>")
    filename_scores_html.write(report_html + "<br>")
    filename_scores_html.write('<img src = "' + classification_report_plot + '" alt ="cfg"><br>')
    filename_scores_html.write("<font size= "'6'"><b> " + "Roc Auc curve figure:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + roc_curve_plot + '" alt ="cfg"><br>')
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Overfit Report:" + "<br>")
    filename_scores_html.write(custom_overfit_report + "<br>")
    filename_scores_html.write("<font size= "'6'"><b> " + "Learning Curve - Overfitting or Underfitting:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + overfitting_plot + '" alt ="cfg">')
    filename_scores_html.close()


print_report(pipeline)
filename = os.path.join(file_dir, """ + model_name2 + """)
joblib.dump(pipeline, filename=filename)\n\n""")
                        elif selected_hp_tuning == "RandomizedSearchCV":
                            f.write("""
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, random_state=0, scoring="accuracy", n_jobs=2,
                                   cv=5)
random_search.fit(X_train, Y_train)
file_dir = os.path.dirname(__file__)

# Calculate the mean and standard deviation of the scores
mean_score = np.mean(random_search.cv_results_['mean_test_score'])
std_score = np.std(random_search.cv_results_['mean_test_score'])

# Get the minimum and maximum scores
min_score = np.min(random_search.cv_results_['mean_test_score'])
max_score = np.max(random_search.cv_results_['mean_test_score'])


pipeline.set_params(**random_search.best_params_)
pipeline.get_params(""" + model_name + """)
pipeline.fit(X_train, Y_train)

# Evaluate the model on the training set
train_acc = accuracy_score(Y_train, pipeline.predict(X_train))
val_acc = accuracy_score(Y_test, pipeline.predict(X_test))
overfit_estimation = train_acc - val_acc

# Generate a learning curve to estimate the overfit
train_sizes, train_scores, test_scores = learning_curve(pipeline, X, Y_all, cv=5, scoring="accuracy", n_jobs=-1,
                                                        train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate the training and testing accuracy with learning curve
train_acc_learning_curve = np.mean(train_scores, axis=1)
val_acc_learning_curve = np.mean(test_scores, axis=1)

# Calculate the overfit gap
overfit_gap_learning_curve = train_acc_learning_curve - val_acc_learning_curve

# Get the last value
last_overfit_gap = overfit_gap_learning_curve[-1]

# Plot the learning curve
plt.grid()
plt.plot(train_sizes, train_acc_learning_curve, label='Training accuracy')
plt.plot(train_sizes, val_acc_learning_curve, label='Testing accuracy')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.xlabel('Training size')
plt.ylabel('Accuracy')
# plt.ylim(0, 1.00)
ax = plt.gca()
ax.set_yticks(np.arange(0, 1.10, 0.05))
plt.title('Learning Curve')

# Plot the overfit gap
plt.twinx()
plt.plot(train_sizes, overfit_gap_learning_curve, color='r', label='Overfit gap')
plt.legend(loc='lower left')
plt.ylabel('Overfit gap')

# Save the plot
plot_image = os.path.join(file_dir, 'overfitting_plot.png')
plt.savefig(plot_image, dpi=250, format='png', bbox_inches='tight')

# Save Pipeline diagram in HTML
set_config(display="diagram")
pipeline_diagram = estimator_html_repr(pipeline)
pipeline_diagram_file = open(os.path.join(file_dir, 'pipeline_diagram.html'), "w", encoding='utf-8')
pipeline_diagram_file.write(pipeline_diagram)

# ###################################
# ######### METRICS #################
# ###################################

# Make classification report image
def show_values(pc, fmt="%.2f", **kw):
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, 
            correct_orientation=False, cmap='RdBu'):

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    lines = classification_report.split('\\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 4)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height,
            correct_orientation, cmap=cmap)


y_true = Y_test
y_pred = pipeline.predict(X_test)
if (target_labels == [1, 0]) or (target_labels == [0, 1]) or (target_labels == [1, 2]) or (target_labels == [2, 1]) or (target_labels == [-1, 1]) or (target_labels == [1, -1]):
    report = metrics.classification_report(y_true, y_pred)
else:
    report = metrics.classification_report(y_true, y_pred, target_names=target_labels)
print(report)


# Plot classification report and save in figure
plot_classification_report(report)
classification_report_plot_name = os.path.join(file_dir, 'test_plot_classif_report.png')
plt.savefig(classification_report_plot_name, dpi=200, format='png', bbox_inches='tight')
# plt.savefig(file_dir + 'test_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')
plt.close()

# Plot Roc curve and save it in figure
metrics.RocCurveDisplay.from_predictions(y_true, y_pred)
roc_curve_plot_name = os.path.join(file_dir, "plot_roc_curve.png")
plt.savefig(roc_curve_plot_name, dpi=200, format='png', bbox_inches='tight')
# plt.savefig(file_dir + 'plot_roc_curve.png', dpi=200, format='png', bbox_inches='tight')
plt.close()

classification_report_plot = os.path.abspath('test_plot_classif_report.png')
roc_curve_plot = os.path.abspath('plot_roc_curve.png')
overfitting_plot = os.path.abspath('overfitting_plot.png')

def print_report(random_search):
    # TXT report
    filename_scores = open(os.path.join(file_dir, """ + model_name3 + """), "w", encoding='utf-8')
    y_true = Y_test
    y_pred = pipeline.predict(X_test)
    # report = metrics.classification_report(y_true, y_pred, target_names=target_labels)
    filename_scores.write("Problem: Classification" + "\\n")
    filename_scores.write("model's name: " """ + model_name + """ + "\\n")
    filename_scores.write(report + "\\n")
    filename_scores.write("Accuracy score: {:0.5f}".format(metrics.accuracy_score(y_true, y_pred)))
    filename_scores.write("\\n")
    filename_scores.write("F1 score: {:0.3f}".format(metrics.f1_score(y_true, y_pred)))
    filename_scores.write("\\n")
    filename_scores.write("Auc score: {:0.3f}".format(metrics.roc_auc_score(Y_test, y_pred)) + "\\n")
    filename_scores.write("Overfit estimation with different {:?f}" + "\\n")
    filename_scores.write("{:0.1f}".format(overfit_estimation) + "\\n")
    filename_scores.write("{:0.2f}".format(overfit_estimation) + "\\n")
    filename_scores.write("{:0.3f}".format(overfit_estimation) + "\\n")
    filename_scores.write("{:0.4f}".format(overfit_estimation) + "\\n")
    filename_scores.write("Learning curve with different {:?f} for the overall mean value"+ "\\n")
    filename_scores.write("{:0.1f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("{:0.2f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("{:0.3f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("{:0.4f}".format(np.mean(overfit_gap_learning_curve)) + "\\n")
    filename_scores.write("Learning curve with different {:?f} for the last value" + "\\n")
    filename_scores.write("{:0.1f}".format(last_overfit_gap) + "\\n")
    filename_scores.write("{:0.2f}".format(last_overfit_gap) + "\\n")
    filename_scores.write("{:0.3f}".format(last_overfit_gap) + "\\n")
    filename_scores.write("{:0.4f}".format(last_overfit_gap) + "\\n")
    filename_scores.close() 
    # HTML report
    report_dict = metrics.classification_report(y_true, y_pred, target_names=target_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_html = report_df.to_html()
    if std_score < 0.1 * mean_score:
        score_consistensy = "The scores are relatively consistent."
    else:
        score_consistensy = "The scores are relatively inconsistent."
    if train_acc > val_acc:
        overfit = "The model may be overfitting."
    else:
        overfit = "The model does not seem to be overfitting."
    custom_dict_report_html = {
        "Problem": ["Classification"],
        "Target Column Name": [" """ + target_column + """ "],
        "Model's Name": [""" + model_name + """],
        "Accuracy Score": ["{:0.5f}".format(metrics.accuracy_score(y_true, y_pred))],
        "Roc Auc curve": ["{:0.3f}".format(metrics.roc_auc_score(Y_test, y_pred))],
        "Mean accuracy score of each tested hyperparameter combination": ["{:0.3f}".format(mean_score)],
        "Range of all accuracy scores of each tested hyperparameter combination":
            ["{:.3f} - {:.3f}".format(min_score, max_score)],
        "Standard Deviation of scores": ["{:0.3f}".format(std_score)],
        "Standard Deviation < 0.1 * Mean Accuracy scores": ["" + score_consistensy + ""]}
    df_custom_report_html = pd.DataFrame(custom_dict_report_html).transpose()
    custom_report_html = df_custom_report_html.to_html(header=False)
    overfit_report = {
        "Overfit Report": ["The Report is based only on Accuracy"],
        "Train set accuracy score of best pipeline": ["{:0.4f}".format(train_acc)],
        "Test set accuracy score of best pipeline": ["{:0.4f}".format(val_acc)],
        "Overfit estimation score of the best pipeline": ["{:0.4f}".format(overfit_estimation)],
        "Learning Curve scores report": ["The Learning Curve is based on Accuracy"],
        "Train set accuracy score of learning curve's last value": ["{:0.2f}".format(train_acc_learning_curve[-1])],
        "Test set accuracy score of learning curve's last value": ["{:0.2f}".format(val_acc_learning_curve[-1])],
        "Overfit gap of learning curve's last value": ["{:0.2f}".format(last_overfit_gap)]}
    df_overfit_report_html = pd.DataFrame(overfit_report).transpose()
    custom_overfit_report = df_overfit_report_html.to_html(header=False)
    filename_scores_html = open(os.path.join(file_dir, """ + model_name4 + """), "w", encoding='utf-8')
    filename_scores_html.write(custom_report_html)
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Classification Report:" + "<br>")
    filename_scores_html.write(report_html + "<br>")
    filename_scores_html.write('<img src = "' + classification_report_plot + '" alt ="cfg"><br>')
    filename_scores_html.write("<font size= "'6'"><b> " + "Roc Auc curve figure:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + roc_curve_plot + '" alt ="cfg"><br>')
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Overfit Report:" + "<br>")
    filename_scores_html.write(custom_overfit_report + "<br>")
    filename_scores_html.write("<font size= "'6'"><b> " + "Learning Curve - Overfitting or Underfitting:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + overfitting_plot + '" alt ="cfg">')
    filename_scores_html.close()


print_report(pipeline)
filename = os.path.join(file_dir, """ + model_name2 + """)
joblib.dump(pipeline, filename=filename)\n\n""")
                    elif machine_learning_problem == "Regression":
                        if selected_hp_tuning == "GridSearchCV" and len(grid_params) > 8:
                            # use RandomSearchCV if the hyper_parameter are greater than 9 and create metrics file
                            # for the pipeline
                            f.write("""
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, random_state=0, scoring="r2", n_jobs=2,
                                cv=5)
random_search.fit(X_train, y_train)
file_dir = os.path.dirname(__file__)

# Get the scores for each combination of parameters
scores = -random_search.cv_results_['mean_test_score']  # Negate the R2 scores to get the correct values

# Calculate the mean, standard deviation, and range of the scores
mean_score = np.mean(scores)
std_score = np.std(scores)
min_score = np.min(scores)
max_score = np.max(scores)


pipeline.set_params(**random_search.best_params_)
pipeline.get_params(""" + model_name + """)
pipeline.fit(X_train, y_train)

# Calculate the mean squared error for train and test set
from sklearn.metrics import r2_score
train_r2 = r2_score(y_train, pipeline.predict(X_train))
val_r2 = r2_score(y_test, pipeline.predict(X_test))
overfit_estimation = train_r2 - val_r2

# Generate a learning curve to estimate the overfit
train_sizes, train_scores, test_scores = learning_curve(pipeline, X, y, cv=5, n_jobs=-1, 
                                                        scoring='r2',
                                                        train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate the training and testing accuracy with learning curve
train_mse_learning_curve = np.mean(train_scores, axis=1)
val_mse_learning_curve = np.mean(test_scores, axis=1)

# Calculate the overfit gap
overfit_gap_learning_curve = train_mse_learning_curve - val_mse_learning_curve

# Get the last value
last_overfit_gap = overfit_gap_learning_curve[-1]

# Plot the learning curve
plt.grid()
plt.plot(train_sizes, train_mse_learning_curve, label='Training error')
plt.plot(train_sizes, val_mse_learning_curve, label='Test error')
plt.legend(loc="upper right")
plt.xlabel('Number of training examples')
plt.ylabel('Mean squared error')
plt.title('Learning Curve')

# Plot the overfit gap
plt.twinx()
plt.plot(train_sizes, overfit_gap_learning_curve, color='r', label='Overfit gap')
plt.legend(loc='lower left')
plt.ylabel('Overfit gap')

# Save the plot
plot_image = os.path.join(file_dir, 'overfitting_plot.png')
plt.savefig(plot_image, dpi=250, format='png', bbox_inches='tight')

# Save Pipeline diagram in HTML
set_config(display="diagram")
pipeline_diagram = estimator_html_repr(pipeline)
pipeline_diagram_file = open(os.path.join(file_dir, 'pipeline_diagram.html'), "w", encoding='utf-8')
pipeline_diagram_file.write(pipeline_diagram)

# ###################################
# ######### METRICS #################
# ###################################
overfitting_plot = os.path.abspath('overfitting_plot.png')


def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    #mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    #median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    # TXT Report
    filename_scores = open(os.path.join(file_dir, """ + model_name3 + """), "w", encoding='utf-8')
    filename_scores.write("Problem: Regression" + "\\n")
    filename_scores.write("model's name: " """ + model_name + """ + "\\n")
    filename_scores.write('explained_variance: ' + str(round(explained_variance, 4)) + "\\n")
    #filename_scores.write('mean_squared_log_error: ' + str(round(mean_squared_log_error, 4)) + "\\n")
    filename_scores.write('r2: ' + str(round(r2, 4)) + "\\n")
    filename_scores.write('MAE: ' + str(round(mean_absolute_error, 4)) + "\\n")
    filename_scores.write('MSE: ' + str(round(mse, 4)) + "\\n")
    filename_scores.write('RMSE: ' + str(round(np.sqrt(mse), 4)) + "\\n")
    filename_scores.close()
    # HTML Report
    if std_score < 0.1 * mean_score:
        score_consistensy = "The R2 values are relatively consistent."
    else:
        score_consistensy = "The R2 values are relatively inconsistent."
    if train_r2 > val_r2:
        overfit = "The model may be overfitting."
    else:
        overfit = "The model does not seem to be overfitting."
    dict_html = {"Problem": ["Regression"],
                 "Target Column Name": [" """ + target_column + """ "],
                 "model's name": [""" + model_name + """],
                 "explained_variance": [str(round(explained_variance, 4))],
                 #"Mean Squared Log Error": [str(round(mean_squared_log_error, 4))],
                 "R2 Score": [str(round(r2, 4))],
                 "Mean Absolute Error": [str(round(mean_absolute_error, 4))],
                 "Mean Squared Error": [str(round(mse, 4))],
                 "Root Mean Squared Error": [str(round(np.sqrt(mse), 4))],
                 "Mean R2": ["{:.3f}".format(mean_score)],
                 "Standard deviation of R2": ["{:.3f}".format(std_score)],
                 "Standard Deviation < 0.1 * Mean R2 scores": ["" + score_consistensy + ""]}
    df_dict_html = pd.DataFrame(dict_html).transpose()
    report_html = df_dict_html.to_html(header=False)
    overfit_report = {
        "Overfit Report": ["The Report is based only on R2 Score"],
        "Overfiting or Not Overfitting": ["" + overfit + ""],
        "Train set R2 score of best pipeline": [str(round(train_r2, 4))],
        "Test set R2 score of best pipeline": [str(round(val_r2, 4))],
        "Overfit estimation score of the best pipeline": [str(round(overfit_estimation, 4))],
        "Learning Curve scores report": ["The Learning Curve is based on R2 Score"],
        "Train set R2 score of learning curve's last value": 
            [str(round(train_mse_learning_curve[-1], 4))],
        "Test set R2 score of learning curve's last value": 
            [str(round(val_mse_learning_curve[-1], 4))],
        "Overfit gap of learning curve's last value": [str(round(last_overfit_gap, 4))]}
    df_overfit_report_html = pd.DataFrame(overfit_report).transpose()
    custom_overfit_report = df_overfit_report_html.to_html(header=False)
    filename_scores_html = open(os.path.join(file_dir, """ + model_name4 + """), "w", encoding='utf-8')
    filename_scores_html.write(report_html + "<br>")
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Overfit Report:" + "<br>")
    filename_scores_html.write(custom_overfit_report + "<br>")
    filename_scores_html.write("<font size= "'6'"><b> " + "Learning Curve - Overfitting or Underfitting:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + overfitting_plot + '" alt ="cfg">')
    filename_scores_html.close()
    

y_true = y_test
y_pred = pipeline.predict(X_test)
regression_results(y_true, y_pred)
filename = os.path.join(file_dir, """ + model_name2 + """)
joblib.dump(pipeline, filename=filename)\n\n""")
                        elif selected_hp_tuning == "GridSearchCV" and len(grid_params) <= 8:
                            # else use GridSearchCv and create metrics file for the pipeline
                            f.write("""
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=5, scoring="r2", cv=5, n_jobs=2)
grid_search.fit(X_train, y_train)


# Get the scores for each combination of parameters
scores = grid_search.cv_results_['mean_test_score']  # Negate the R2 scores to get the correct values

# Calculate the mean, standard deviation, and range of the scores
mean_score = np.mean(scores)
std_score = np.std(scores)
min_score = np.min(scores)
max_score = np.max(scores)

file_dir = os.path.dirname(__file__)
pipeline.set_params(**grid_search.best_params_)
pipeline.get_params(""" + model_name + """)
pipeline.fit(X_train, y_train)

# Calculate the mean squared error for train and test set
from sklearn.metrics import r2_score
train_r2 = r2_score(y_train, pipeline.predict(X_train))
val_r2 = r2_score(y_test, pipeline.predict(X_test))
overfit_estimation = train_r2 - val_r2

# Generate a learning curve to estimate the overfit
train_sizes, train_scores, test_scores = learning_curve(pipeline, X, y, cv=5, n_jobs=-1, 
                                                        scoring='r2',
                                                        train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate the training and testing accuracy with learning curve
train_mse_learning_curve = np.mean(train_scores, axis=1)
val_mse_learning_curve = np.mean(test_scores, axis=1)

# Calculate the overfit gap
overfit_gap_learning_curve = train_mse_learning_curve - val_mse_learning_curve

# Get the last value
last_overfit_gap = overfit_gap_learning_curve[-1]

# Plot the learning curve
plt.grid()
plt.plot(train_sizes, train_mse_learning_curve, label='Training error')
plt.plot(train_sizes, val_mse_learning_curve, label='Test error')
plt.legend(loc="upper right")
plt.xlabel('Number of training examples')
plt.ylabel('Mean squared error')
plt.title('Learning Curve')

# Plot the overfit gap
plt.twinx()
plt.plot(train_sizes, overfit_gap_learning_curve, color='r', label='Overfit gap')
plt.legend(loc='lower left')
plt.ylabel('Overfit gap')

# Save the plot
plot_image = os.path.join(file_dir, 'overfitting_plot.png')
plt.savefig(plot_image, dpi=250, format='png', bbox_inches='tight')

# Save Pipeline diagram in HTML
set_config(display="diagram")
pipeline_diagram = estimator_html_repr(pipeline)
pipeline_diagram_file = open(os.path.join(file_dir, 'pipeline_diagram.html'), "w", encoding='utf-8')
pipeline_diagram_file.write(pipeline_diagram)

# ###################################
# ######### METRICS #################
# ###################################
overfitting_plot = os.path.abspath('overfitting_plot.png')


def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
   # mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    #median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    # TXT Report
    filename_scores = open(os.path.join(file_dir, """ + model_name3 + """), "w", encoding='utf-8')
    filename_scores.write("Problem: Regression" + "\\n")
    filename_scores.write("model's name: " """ + model_name + """ + "\\n")
    filename_scores.write('explained_variance: ' + str(round(explained_variance, 4)) + "\\n")
    #filename_scores.write('mean_squared_log_error: ' + str(round(mean_squared_log_error, 4)) + "\\n")
    filename_scores.write('r2: ' + str(round(r2, 4)) + "\\n")
    filename_scores.write('MAE: ' + str(round(mean_absolute_error, 4)) + "\\n")
    filename_scores.write('MSE: ' + str(round(mse, 4)) + "\\n")
    filename_scores.write('RMSE: ' + str(round(np.sqrt(mse), 4)) + "\\n")
    filename_scores.close()
    # HTML Report
    if std_score < 0.1 * mean_score:
        score_consistensy = "The R2 values are relatively consistent."
    else:
        score_consistensy = "The R2 values are relatively inconsistent."
    if train_r2 > val_r2:
        overfit = "The model may be overfitting."
    else:
        overfit = "The model does not seem to be overfitting."
    dict_html = {"Problem": ["Regression"],
                 "Target Column Name": [" """ + target_column + """ "],
                 "model's name": [""" + model_name + """],
                 "explained_variance": [str(round(explained_variance, 4))],
                 #"Mean Squared Log Error": [str(round(mean_squared_log_error, 4))],
                 "R2 Score": [str(round(r2, 4))],
                 "Mean Absolute Error": [str(round(mean_absolute_error, 4))],
                 "Mean Squared Error": [str(round(mse, 4))],
                 "Root Mean Squared Error": [str(round(np.sqrt(mse), 4))],
                 "Mean R2": ["{:.3f}".format(mean_score)],
                 "Standard deviation of R2": ["{:.3f}".format(std_score)],
                 "Standard Deviation < 0.1 * Mean R2 scores": ["" + score_consistensy + ""]}
    df_dict_html = pd.DataFrame(dict_html).transpose()
    report_html = df_dict_html.to_html(header=False)
    overfit_report = {
        "Overfit Report": ["The Report is based only on R2 Score"],
        "Overfiting or Not Overfitting": ["" + overfit + ""],
        "Train set R2 score of best pipeline": [str(round(train_r2, 4))],
        "Test set R2 score of best pipeline": [str(round(val_r2, 4))],
        "Overfit estimation score of the best pipeline": [str(round(overfit_estimation, 4))],
        "Learning Curve scores report": ["The Learning Curve is based on R2 Score"],
        "Train set R2 score of learning curve's last value": 
            [str(round(train_mse_learning_curve[-1], 4))],
        "Test set R2 score of learning curve's last value": 
            [str(round(val_mse_learning_curve[-1], 4))],
        "Overfit gap of learning curve's last value": [str(round(last_overfit_gap, 4))]}
    df_overfit_report_html = pd.DataFrame(overfit_report).transpose()
    custom_overfit_report = df_overfit_report_html.to_html(header=False)
    filename_scores_html = open(os.path.join(file_dir, """ + model_name4 + """), "w", encoding='utf-8')
    filename_scores_html.write(report_html + "<br>")
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Overfit Report:" + "<br>")
    filename_scores_html.write(custom_overfit_report + "<br>")
    filename_scores_html.write("<font size= "'6'"><b> " + "Learning Curve - Overfitting or Underfitting:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + overfitting_plot + '" alt ="cfg">')
    filename_scores_html.close()


y_true = y_test
y_pred = pipeline.predict(X_test)
regression_results(y_true, y_pred)
filename = os.path.join(file_dir, """ + model_name2 + """)
joblib.dump(pipeline, filename=filename)\n\n""")
                        elif selected_hp_tuning == "RandomizedSearchCV":
                            f.write("""
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, random_state=0, scoring="r2", n_jobs=-1,
                                   cv=5)
random_search.fit(X_train, y_train)
print(random_search.best_estimator_)
print(random_search.best_score_)
print(random_search.best_params_)


# Get the scores for each combination of parameters
scores = random_search.cv_results_['mean_test_score']  # Negate the R2 scores to get the correct values

# Calculate the mean, standard deviation, and range of the scores
mean_score = np.mean(scores)
std_score = np.std(scores)
min_score = np.min(scores)
max_score = np.max(scores)


file_dir = os.path.dirname(__file__)
pipeline.set_params(**random_search.best_params_)
pipeline.get_params(""" + model_name + """)
pipeline.fit(X_train, y_train)

# Calculate the mean squared error for train and test set
from sklearn.metrics import r2_score
train_r2 = r2_score(y_train, pipeline.predict(X_train))
val_r2 = r2_score(y_test, pipeline.predict(X_test))
overfit_estimation = train_r2 - val_r2

# Generate a learning curve to estimate the overfit
train_sizes, train_scores, test_scores = learning_curve(pipeline, X, y, cv=5, n_jobs=-1, 
                                                        scoring='r2',
                                                        train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate the training and testing accuracy with learning curve
train_mse_learning_curve = np.mean(train_scores, axis=1)
val_mse_learning_curve = np.mean(test_scores, axis=1)

# Calculate the overfit gap
overfit_gap_learning_curve = train_mse_learning_curve - val_mse_learning_curve

# Get the last value
last_overfit_gap = overfit_gap_learning_curve[-1]

# Plot the learning curve
plt.grid()
plt.plot(train_sizes, train_mse_learning_curve, label='Training error')
plt.plot(train_sizes, val_mse_learning_curve, label='Test error')
plt.legend(loc="upper right")
plt.xlabel('Number of training examples')
plt.ylabel('Mean squared error')
plt.title('Learning Curve')

# Plot the overfit gap
plt.twinx()
plt.plot(train_sizes, overfit_gap_learning_curve, color='r', label='Overfit gap')
plt.legend(loc='lower left')
plt.ylabel('Overfit gap')

# Save the plot
plot_image = os.path.join(file_dir, 'overfitting_plot.png')
plt.savefig(plot_image, dpi=250, format='png', bbox_inches='tight')

# Save Pipeline diagram in HTML
set_config(display="diagram")
pipeline_diagram = estimator_html_repr(pipeline)
pipeline_diagram_file = open(os.path.join(file_dir, 'pipeline_diagram.html'), "w", encoding='utf-8')
pipeline_diagram_file.write(pipeline_diagram)

# ###################################
# ######### METRICS #################
# ###################################

overfitting_plot = os.path.abspath('overfitting_plot.png')


def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    #mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
   # median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    # TXT Report
    filename_scores = open(os.path.join(file_dir, """ + model_name3 + """), "w", encoding='utf-8')
    filename_scores.write("Problem: Regression" + "\\n")
    filename_scores.write("model's name: " """ + model_name + """ + "\\n")
    filename_scores.write('explained_variance: ' + str(round(explained_variance, 4)) + "\\n")
    #filename_scores.write('mean_squared_log_error: ' + str(round(mean_squared_log_error, 4)) + "\\n")
    filename_scores.write('r2: ' + str(round(r2, 4)) + "\\n")
    filename_scores.write('MAE: ' + str(round(mean_absolute_error, 4)) + "\\n")
    filename_scores.write('MSE: ' + str(round(mse, 4)) + "\\n")
    filename_scores.write('RMSE: ' + str(round(np.sqrt(mse), 4)) + "\\n")
    filename_scores.close()
    # HTML Report
    if std_score < 0.1 * mean_score:
        score_consistensy = "The R2 values are relatively consistent."
    else:
        score_consistensy = "The R2 values are relatively inconsistent."
    if train_r2 > val_r2:
        overfit = "The model may be overfitting."
    else:
        overfit = "The model does not seem to be overfitting."
    dict_html = {"Problem": ["Regression"],
                 "Target Column Name": [" """ + target_column + """ "],
                 "model's name": [""" + model_name + """],
                 "explained_variance": [str(round(explained_variance, 4))],
                 #"Mean Squared Log Error": [str(round(mean_squared_log_error, 4))],
                 "R2 Score": [str(round(r2, 4))],
                 "Mean Absolute Error": [str(round(mean_absolute_error, 4))],
                 "Mean Squared Error": [str(round(mse, 4))],
                 "Root Mean Squared Error": [str(round(np.sqrt(mse), 4))],
                 "Mean R2": ["{:.3f}".format(mean_score)],
                 "Standard deviation of R2": ["{:.3f}".format(std_score)],
                 "Standard Deviation < 0.1 * Mean R2 scores": ["" + score_consistensy + ""]}
    df_dict_html = pd.DataFrame(dict_html).transpose()
    report_html = df_dict_html.to_html(header=False)
    overfit_report = {
        "Overfit Report": ["The Report is based only on R2 score"],
        "Overfiting or Not Overfitting": ["" + overfit + ""],
        "Train set R2 score of best pipeline": [str(round(train_r2, 4))],
        "Test set R2 score of best pipeline": [str(round(val_r2, 4))],
        "Overfit estimation score of the best pipeline": [str(round(overfit_estimation, 4))],
        "Learning Curve scores report": ["The Learning Curve is based on R2 Score"],
        "Train set R2 score of learning curve's last value": [str(round(train_mse_learning_curve, 4))],
        "Test set R2 score of learning curve's last value": [str(round(val_mse_learning_curve, 4))],
        "Overfit gap of learning curve's last value": [str(round(last_overfit_gap, 4))]}
    df_overfit_report_html = pd.DataFrame(overfit_report).transpose()
    custom_overfit_report = df_overfit_report_html.to_html(header=False)
    filename_scores_html = open(os.path.join(file_dir, """ + model_name4 + """), "w", encoding='utf-8')
    filename_scores_html.write(report_html + "<br>")
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Overfit Report:" + "<br>")
    filename_scores_html.write(custom_overfit_report + "<br>")
    filename_scores_html.write("<font size= "'6'"><b> " + "Learning Curve - Overfitting or Underfitting:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + overfitting_plot + '" alt ="cfg">')
    filename_scores_html.close()


y_true = y_test
y_pred = pipeline.predict(X_test)
regression_results(y_true, y_pred)
filename = os.path.join(file_dir, """ + model_name2 + """)
joblib.dump(pipeline, filename=filename)\n\n""")
                    f.write("""
data_preprocessor = os.path.join(file_dir, 'Data_preprocessor.joblib')
joblib.dump(pipeline.named_steps["preprocessor"], filename=data_preprocessor)""" + "\n")
                    if categorical_features_true[0] == 1:
                        f.write("""
preprocessor = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
ohe_categories = preprocessor.named_steps["OneHotEncoder"].categories_
new_ohe_features = [f"{col}__{val}" for col, vals in zip(cat_features, ohe_categories) for val in vals]
all_features = num_features + new_ohe_features
all_features_file = open(os.path.join(file_dir, 'all_features'), 'wb')
dill.dump(all_features, all_features_file)""" + "\n")
                    else:
                        f.write("""
all_features = num_features
all_features_file = open(os.path.join(file_dir, 'all_features'), 'wb')
dill.dump(all_features, all_features_file)""" + "\n")
                    # take the name of the fs
                    point_dim_redu = []
                    only_dim_redu = []
                    for point in dim_redu_pipeline_points:
                        point_dim_redu.append(str(point))
                    if len(fs_pipeline_points) == 1 and len(point_dim_redu) == 1:
                        only_dim_redu.append("yes")
                    else:
                        only_dim_redu.append("no")
                    fs_name = ""
                    if fs_pipeline_points and (only_dim_redu[0] == "no"):
                        for key, value in conf.model_names_for_pipeline_dict.items():
                            if fs_point == key:
                                if type(value) == dict:
                                    for item1, item2 in value.items():
                                        fs_name = repr(item1)
                        f.write("fs = pipeline.named_steps[" + fs_name + "]\n")
                        f.write("fs_names = np.array(all_features)\n")
                        f.write("selected_features = fs_names[fs.get_support()]\n")
                        f.write("""
sel_fea = []
for feature in selected_features:
    sel_fea.append(feature.split('__')[0])
print(sel_fea)
selected_features_file = open(os.path.join(file_dir, 'selected_features'), "wb")
dill.dump(selected_features, selected_features_file)
\n""")
                    # change the path
                    if categorical_features_true[0] == 1:
                        f.write("""
categorical_names = {}
for col in cat_features:
    if col not in categorical_features_with_numerical_values:
        categorical_names[X_train.columns.get_loc(col)] = [new_col.split("__")[1]
                                                           for new_col in new_ohe_features
                                                           if new_col.split("__")[0] == col]
print(categorical_names)
# save categorical_names
categorical_names_file = open(os.path.join(file_dir, 'categorical_names.pkl'), "wb")
dill.dump(categorical_names, categorical_names_file)

full_categorical_names = {}
for col in cat_features:
    full_categorical_names[X_train.columns.get_loc(col)] = [new_col.split("__")[1]
                                                       for new_col in new_ohe_features
                                                       if new_col.split("__")[0] == col]
print(full_categorical_names)
full_categorical_names_file = open(os.path.join(file_dir, 'full_categorical_names.pkl'), "wb")
dill.dump(full_categorical_names, full_categorical_names_file)

# #############################################
# ######### LIME EXPLANATIONS #################
# #############################################

def convert_to_lime_format(X, categorical_names, col_names=None, invert=False):
    # """"""Converts data with categorical values as string into the right format
    # for LIME, with categorical values as integers labels.
    #
    # It takes categorical_names, the same dictionary that has to be passed
    # to LIME to ensure consistency.
    #
    # col_names and invert allow to rebuild the original dataFrame from
    # a numpy array in LIME format to be passed to a Pipeline or sklearn
    # OneHotEncoder""""""
    # If the data isn't a dataframe, we need to be able to build it
    if not isinstance(X, pd.DataFrame):
        X_lime = pd.DataFrame(X, columns=col_names)
    else:
        X_lime = X.copy()
    for k, v in categorical_names.items():
        if not invert:
            label_map = {
                str_label: int_label for int_label, str_label in enumerate(v)
            }
        else:
            label_map = {
                int_label: str_label for int_label, str_label in enumerate(v)
            }
        X_lime.iloc[:, k] = X_lime.iloc[:, k].map(label_map)
    return X_lime


print(convert_to_lime_format(X_train, categorical_names).head())


def convert_to_imputed_lime(X):
    from sklearn.impute import SimpleImputer
    lime_data = convert_to_lime_format(X, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(lime_data)
    imputed_lime_data = pd.DataFrame(imp_mean.fit_transform(lime_data), columns=X.columns,
                                     index=lime_data.index)
    return imputed_lime_data


imputed_lime_X_train = convert_to_imputed_lime(X_train)
 \n""")
                        if machine_learning_problem == "Classification":
                            f.write("""
lime_explainer = lime.lime_tabular.LimeTabularExplainer(imputed_lime_X_train.values,
                                      mode="classification",
                                      class_names=target_labels,
                                      feature_names=X_train.columns.tolist(),
                                      categorical_names=categorical_names,
                                      categorical_features=categorical_names.keys(),
                                      discretize_continuous=False,
                                      kernel_width=5,
                                      random_state=42)


imputed_lime_X_test = convert_to_imputed_lime(X_test)


lime_explainer_file = open(os.path.join(file_dir, 'lime_explainer_file'), "wb")
dill.dump(lime_explainer, lime_explainer_file)
             """)
                        elif machine_learning_problem == "Regression":
                            f.write("""
lime_explainer = lime.lime_tabular.LimeTabularExplainer(imputed_lime_X_train.values,
                                      mode="regression",
                                      feature_names=X_train.columns.tolist(),
                                      class_names=" """ + target_column + """ ",
                                      categorical_names=categorical_names,
                                      categorical_features=categorical_names.keys(),
                                      discretize_continuous=False,
                                      kernel_width=5,
                                      random_state=42)


imputed_lime_X_test = convert_to_imputed_lime(X_test)


# save explanation to a dill file
with open('lime_explainer_file', 'wb') as f:
    dill.dump(lime_explainer, f)  
                 """)
                        # shap values
                        if (shap_explainer_type[0] == "TreeExplainer") and (len(fs_pipeline_points) != 0):
                            f.write("""
new_selected_features = []
for feature in selected_features:
    new_selected_features.append(feature.split("__", 1)[0])

from sklearn.impute import SimpleImputer


def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(j_data)
    custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
    if len(new_selected_features) > 0:
        return custom_data[new_selected_features]
    else:
        data = x
        ohe_data = pd.DataFrame(pipeline.named_steps["preprocessor"].transform(data), columns=all_features)
        return ohe_data


# #############################################
# ######### SHAP VALUES ############
# #############################################

shap_data = shap_and_eli5_custom_format(X_train)

shap.initjs()
shap_explainer = shap.TreeExplainer(pipeline.named_steps[""" + model_name + """])
shap_explainer_file = open(os.path.join(file_dir, 'Tree_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "TreeExplainer") and (len(fs_pipeline_points) == 0):
                            f.write("""
from sklearn.impute import SimpleImputer


def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(j_data)
    custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
    data = x
    ohe_data = pd.DataFrame(pipeline.named_steps["preprocessor"].transform(data), columns=all_features)
    return ohe_data


# #############################################
# ######### SHAP VALUES ############
# #############################################
shap_data = shap_and_eli5_custom_format(X_train)

shap.initjs()
shap_explainer = shap.TreeExplainer(pipeline.named_steps[""" + model_name + """])
shap_explainer_file = open(os.path.join(file_dir, 'Tree_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "LinearExplainer") and (len(fs_pipeline_points) != 0):
                            f.write("""
new_selected_features = []
for feature in selected_features:
    new_selected_features.append(feature.split("__", 1)[0])

from sklearn.impute import SimpleImputer


def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(j_data)
    custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
    if len(new_selected_features) > 0:
        return custom_data[new_selected_features]
    else:
        data = x
        ohe_data = pd.DataFrame(pipeline.named_steps["preprocessor"].transform(data), columns=all_features)
        return ohe_data


# #############################################
# ######### SHAP VALUES ############
# #############################################
shap_data = shap_and_eli5_custom_format(X_train)

shap.initjs()
shap_explainer = shap.LinearExplainer(pipeline.named_steps[""" + model_name + """], masker = shap_data)
shap_explainer_file = open(os.path.join(file_dir, 'Linear_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "LinearExplainer") and (len(fs_pipeline_points) == 0):
                            f.write("""
from sklearn.impute import SimpleImputer


def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(j_data)
    custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
    data = x
    ohe_data = pd.DataFrame(pipeline.named_steps["preprocessor"].transform(data), columns=all_features)
    return ohe_data


# #############################################
# ######### SHAP VALUES ############
# #############################################
shap_data = shap_and_eli5_custom_format(X_train)

shap.initjs()
shap_explainer = shap.LinearExplainer(pipeline.named_steps[""" + model_name + """], masker = shap_data)
shap_explainer_file = open(os.path.join(file_dir, 'Linear_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "KernelExplainer") and \
                                (machine_learning_problem == "Classification") and (len(fs_pipeline_points) != 0):
                            f.write("""
new_selected_features = []
for feature in selected_features:
    new_selected_features.append(feature.split("__", 1)[0])

from sklearn.impute import SimpleImputer


def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(j_data)
    custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
    if len(new_selected_features) > 0:
        return custom_data[new_selected_features]
    else:
        data = x
        ohe_data = pd.DataFrame(pipeline.named_steps["preprocessor"].transform(data), columns=all_features)
        return ohe_data


# #############################################
# ######### SHAP VALUES ############
# #############################################
shap_data = shap_and_eli5_custom_format(X_train)
shap_data_summary = shap.kmeans(shap_data, 10)

shap.initjs()
shap_explainer = shap.KernelExplainer(pipeline.named_steps[""" + model_name + """].predict_proba, shap_data_summary)

# save shap values explainer
# with open("shap_explainer", "wb") as f:
#     dill.dump(shap_explainer, f)
shap_explainer_file = open(os.path.join(file_dir, 'Kernel_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "KernelExplainer") and \
                                (machine_learning_problem == "Classification") and (len(fs_pipeline_points) == 0):
                            f.write("""
from sklearn.impute import SimpleImputer


def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(j_data)
    custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
    data = x
    ohe_data = pd.DataFrame(pipeline.named_steps["preprocessor"].transform(data), columns=all_features)
    return ohe_data


# #############################################
# ######### SHAP VALUES ############
# #############################################
shap_data = shap_and_eli5_custom_format(X_train)
shap_data_summary = shap.kmeans(shap_data, 10)

shap.initjs()
shap_explainer = shap.KernelExplainer(pipeline.named_steps[""" + model_name + """].predict_proba, shap_data_summary)

# save shap values explainer
# with open("shap_explainer", "wb") as f:
#     dill.dump(shap_explainer, f)
shap_explainer_file = open(os.path.join(file_dir, 'Kernel_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "KernelExplainer") and (
                                machine_learning_problem == "Regression") \
                                and (len(fs_pipeline_points) != 0):
                            f.write("""
new_selected_features = []
for feature in selected_features:
    new_selected_features.append(feature.split("__", 1)[0])

from sklearn.impute import SimpleImputer


def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(j_data)
    custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
    if len(new_selected_features) > 0:
        return custom_data[new_selected_features]
    else:
        data = x
        ohe_data = pd.DataFrame(pipeline.named_steps["preprocessor"].transform(data), columns=all_features)
        return ohe_data


# #############################################
# ######### SHAP VALUES ############
# #############################################
shap_data = shap_and_eli5_custom_format(X_train)
shap_data_summary = shap.kmeans(shap_data, 10)

shap.initjs()
shap_explainer = shap.KernelExplainer(pipeline.named_steps[""" + model_name + """].predict, shap_data_summary)

# save shap values explainer
# with open("shap_explainer", "wb") as f:
#     dill.dump(shap_explainer, f)
shap_explainer_file = open(os.path.join(file_dir, 'Kernel_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "KernelExplainer") and (
                                machine_learning_problem == "Regression") \
                                and (len(fs_pipeline_points) == 0):
                            f.write("""
from sklearn.impute import SimpleImputer


def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(j_data)
    custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
    data = x
    ohe_data = pd.DataFrame(pipeline.named_steps["preprocessor"].transform(data), columns=all_features)
    return ohe_data


# #############################################
# ######### SHAP VALUES ############
# #############################################
shap_data = shap_and_eli5_custom_format(X_train)
shap_data_summary = shap.kmeans(shap_data, 10)

shap.initjs()
shap_explainer = shap.KernelExplainer(pipeline.named_steps[""" + model_name + """].predict, shap_data_summary)

# save shap values explainer
# with open("shap_explainer", "wb") as f:
#     dill.dump(shap_explainer, f)
shap_explainer_file = open(os.path.join(file_dir, 'Kernel_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                    else:
                        f.write("""
categorical_names = {}

print("There are not categorical features")
# save categorical_names
categorical_names_file = open(os.path.join(file_dir, 'categorical_names.pkl'), "wb")
dill.dump(categorical_names, categorical_names_file)

full_categorical_names = {}
print("There are not categorical features with numerical values")
full_categorical_names_file = open(os.path.join(file_dir, 'full_categorical_names.pkl'), "wb")
dill.dump(full_categorical_names, full_categorical_names_file)
 \n""")
                        if machine_learning_problem == "Classification":
                            f.write("""
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                      mode="classification",
                                      class_names=target_labels,
                                      feature_names=X_train.columns.tolist(),
                                      discretize_continuous=False,
                                      kernel_width=5,
                                      random_state=42)


lime_explainer_file = open(os.path.join(file_dir, 'lime_explainer_file'), "wb")
dill.dump(lime_explainer, lime_explainer_file)
             """)
                        elif machine_learning_problem == "Regression":
                            f.write("""
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                      mode="regression",
                                      feature_names=X_train.columns.tolist(),
                                      class_names=" """ + target_column + """ ",
                                      discretize_continuous=False,
                                      kernel_width=5,
                                      random_state=42)

# save explanation to a dill file
with open('lime_explainer_file', 'wb') as f:
    dill.dump(lime_explainer, f)  
                 """)
                        # shap values
                        if (shap_explainer_type[0] == "TreeExplainer") and (len(fs_pipeline_points) != 0):
                            f.write("""
new_selected_features = []
for feature in selected_features:
    new_selected_features.append(feature.split("__", 1)[0])

# #############################################
# ######### SHAP VALUES ############
# #############################################
shap_data = X_train

shap.initjs()
shap_explainer = shap.TreeExplainer(pipeline.named_steps[""" + model_name + """])
shap_explainer_file = open(os.path.join(file_dir, 'Tree_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "TreeExplainer") and (len(fs_pipeline_points) == 0):
                            f.write("""
shap_data = X_train

# shap values
shap.initjs()
shap_explainer = shap.TreeExplainer(pipeline.named_steps[""" + model_name + """])
shap_explainer_file = open(os.path.join(file_dir, 'Tree_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "LinearExplainer") and (len(fs_pipeline_points) != 0):
                            f.write("""
new_selected_features = []
for feature in selected_features:
    new_selected_features.append(feature.split("__", 1)[0])

# #############################################
# ######### SHAP VALUES ############
# #############################################

shap_data = X_train

shap.initjs()
shap_explainer = shap.LinearExplainer(pipeline.named_steps[""" + model_name + """], masker= shap_data)
shap_explainer_file = open(os.path.join(file_dir, 'Linear_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "LinearExplainer") and (len(fs_pipeline_points) == 0):
                            f.write("""
# #############################################
# ######### SHAP VALUES ############
# #############################################

shap_data = X_train

shap.initjs()
shap_explainer = shap.LinearExplainer(pipeline.named_steps[""" + model_name + """], masker= shap_data)
shap_explainer_file = open(os.path.join(file_dir, 'Linear_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "KernelExplainer") and \
                                (machine_learning_problem == "Classification") and (len(fs_pipeline_points) != 0):
                            f.write("""
new_selected_features = []
for feature in selected_features:
    new_selected_features.append(feature.split("__", 1)[0])

# #############################################
# ######### SHAP VALUES ############
# #############################################

shap_data = X_train
shap_data_summary = shap.kmeans(shap_data, 10)

shap.initjs()
shap_explainer = shap.KernelExplainer(pipeline.named_steps[""" + model_name + """].predict_proba, shap_data_summary)

# save shap values explainer
# with open("shap_explainer", "wb") as f:
#     dill.dump(shap_explainer, f)
shap_explainer_file = open(os.path.join(file_dir, 'Kernel_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "KernelExplainer") and \
                                (machine_learning_problem == "Classification") and (len(fs_pipeline_points) == 0):
                            f.write("""
# #############################################
# ######### SHAP VALUES ############
# #############################################

shap_data = X_train
shap_data_summary = shap.kmeans(shap_data, 10)

shap.initjs()
shap_explainer = shap.KernelExplainer(pipeline.named_steps[""" + model_name + """].predict_proba, shap_data_summary)

shap_explainer_file = open(os.path.join(file_dir, 'Kernel_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "KernelExplainer") and (
                                machine_learning_problem == "Regression") \
                                and (len(fs_pipeline_points) != 0):
                            f.write("""
new_selected_features = []
for feature in selected_features:
    new_selected_features.append(feature.split("__", 1)[0])

# #############################################
# ######### SHAP VALUES #######################
# #############################################

shap_data = X_train
shap_data_summary = shap.kmeans(shap_data, 10)

shap.initjs()
shap_explainer = shap.KernelExplainer(pipeline.named_steps[""" + model_name + """].predict, shap_data_summary)

shap_explainer_file = open(os.path.join(file_dir, 'Kernel_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n""")
                        elif (shap_explainer_type[0] == "KernelExplainer") and (
                                machine_learning_problem == "Regression") \
                                and (len(fs_pipeline_points) == 0):
                            f.write("""

# #############################################
# ######### SHAP VALUES #######################
# #############################################

shap_data = X_train
shap_data_summary = shap.kmeans(shap_data, 10)

shap.initjs()
shap_explainer = shap.KernelExplainer(pipeline.named_steps[""" + model_name + """].predict, shap_data_summary)

shap_explainer_file = open(os.path.join(file_dir, 'Kernel_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)\n
""")
                    if machine_learning_problem == "Classification":
                        if categorical_features_true[0] == 1:
                            f.write("""
from sklearn.impute import SimpleImputer
    
    
def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(j_data)
    custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
    if len(new_selected_features) > 0:
        return custom_data[new_selected_features]
    else:
        data = x
        ohe_data = pd.DataFrame(pipeline.named_steps["preprocessor"].transform(data), columns=all_features)
        return ohe_data

# #############################################
# ######### Permutation Importance ############
# #############################################
    
try:
    import eli5
    from eli5.sklearn import PermutationImportance
    eli5_data = shap_and_eli5_custom_format(X_test)
    permuter = PermutationImportance(pipeline.named_steps[""" + model_name + """],
                                     scoring='accuracy', cv='prefit', n_iter=2, random_state=42)
    permuter.fit(eli5_data, Y_test)
    # explain the predictions of the model using permutation importance
    if len(sel_fea) != 0:
        explanation = eli5.explain_weights(permuter, top=None, feature_names=selected_features)
    else:
        explanation = eli5.explain_weights(permuter, top=None, feature_names=all_features)
    # format the explanation as an HTML string
    html_permutation_importance = eli5.formatters.html.format_as_html(explanation)
    permutation_file = open(os.path.join(file_dir, 'main_explainable_folder', 'eli5_folder',
                                         'permutation_importance.html'), "w", encoding='utf-8')
    permutation_file.write(html_permutation_importance)

except:
    # code to execute in case of any error
    print("An error occurred.")
    print("This model does not support ELI5 library.")""")
                        else:
                            f.write("""
# #############################################
# ######### Permutation Importance ############
# #############################################


try:
    import eli5
    from eli5.sklearn import PermutationImportance
    eli5_data = pipeline.named_steps["preprocessor"].transform(X_test)
    permuter = PermutationImportance(pipeline.named_steps[""" + model_name + """],
                                     scoring='accuracy', cv='prefit', n_iter=2, random_state=42)
    permuter.fit(eli5_data, Y_test)
    # explain the predictions of the model using permutation importance
    if len(sel_fea) != 0:
        explanation = eli5.explain_weights(permuter, top=None, feature_names=selected_features)
    else:
        explanation = eli5.explain_weights(permuter, top=None, feature_names=all_features)
    # format the explanation as an HTML string
    html_permutation_importance = eli5.formatters.html.format_as_html(explanation)
    permutation_file = open(os.path.join(file_dir, 'main_explainable_folder', 'eli5_folder',
                                         'permutation_importance.html'), "w", encoding='utf-8')
    permutation_file.write(html_permutation_importance)

except:
    # code to execute in case of any error
    print("An error occurred.")
    print("This model does not support ELI5 library.")""")
                    else:
                        if categorical_features_true[0] == 1:
                            f.write("""
from sklearn.impute import SimpleImputer


def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(j_data)
    custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
    if len(new_selected_features) > 0:
        return custom_data[new_selected_features]
    else:
        data = x
        ohe_data = pd.DataFrame(pipeline.named_steps["preprocessor"].transform(data), columns=all_features)
        return ohe_data

# #############################################
# ######### Permutation Importance ############
# #############################################


try:
    import eli5
    from eli5.sklearn import PermutationImportance
    eli5_data = shap_and_eli5_custom_format(X_test)
    permuter = PermutationImportance(pipeline.named_steps[""" + model_name + """],
                                     scoring='r2', cv='prefit', n_iter=2, random_state=42)
    permuter.fit(eli5_data, y_test)
    # explain the predictions of the model using permutation importance
    if len(sel_fea) != 0:
        explanation = eli5.explain_weights(permuter, top=None, feature_names=selected_features)
    else:
        explanation = eli5.explain_weights(permuter, top=None, feature_names=all_features)
    # format the explanation as an HTML string
    html_permutation_importance = eli5.formatters.html.format_as_html(explanation)
    permutation_file = open(os.path.join(file_dir, 'main_explainable_folder', 'eli5_folder',
                                         'permutation_importance.html'), "w", encoding='utf-8')
    permutation_file.write(html_permutation_importance)

except:
    # code to execute in case of any error
    print("An error occurred.")
    print("This model does not support ELI5 library.")""")
                        else:
                            f.write("""
# #############################################
# ######### Permutation Importance ############
# #############################################


try:
    import eli5
    from eli5.sklearn import PermutationImportance
    eli5_data = pipeline.named_steps["preprocessor"].transform(X_test)
    permuter = PermutationImportance(pipeline.named_steps[""" + model_name + """],
                                     scoring='r2', cv='prefit', n_iter=2, random_state=42)
    permuter.fit(eli5_data, y_test)
    # explain the predictions of the model using permutation importance
    if len(sel_fea) != 0:
        explanation = eli5.explain_weights(permuter, top=None, feature_names=selected_features)
    else:
        explanation = eli5.explain_weights(permuter, top=None, feature_names=all_features)
    # format the explanation as an HTML string
    html_permutation_importance = eli5.formatters.html.format_as_html(explanation)
    permutation_file = open(os.path.join(file_dir, 'main_explainable_folder', 'eli5_folder',
                                         'permutation_importance.html'), "w", encoding='utf-8')
    permutation_file.write(html_permutation_importance)

except:
    # code to execute in case of any error
    print("An error occurred.")
    print("This model does not support ELI5 library.")""")
                    # ##########################################
                    # creates folders for each explainable model
                    # ##########################################
                    explainable_folder_main = "main_explainable_folder"
                    os.mkdir(os.path.join("" + main_folder, "" + model_name_only + "" + "_Pipeline", ""
                                          + explainable_folder_main + ""))
                    explainable_folders = ["eli5_folder", "lime_folder", "shap_values_folder"]
                    for folder in explainable_folders:
                        os.mkdir(
                            os.path.join("" + main_folder, "" + model_name_only + "" + "_Pipeline", ""
                                         + explainable_folder_main + "", "" + folder + "")
                        )
                    # ######################################
                    # creates a batch file for each pipeline
                    # ######################################
                    batch_file = open(os.path.join("" + main_folder,
                                                   "" + model_name_only + "" + "_Pipeline", "" + model_name_only + ""
                                                   + "_pipeline_run.bat"), "w+")
                    batch_file.write("call " + "C:/Users/micha/anaconda3/Scripts/activate.bat" + "\n")
                    batch_file.write("start " + f.name + "\n")
                    # b.write("pause")
                    # #################################################
                    # create an explanation text file for each pipeline
                    # #################################################
                    explanation = open(os.path.join("" + main_folder,
                                                    "" + model_name_only + "" + "_Pipeline", "" + model_name_only + ""
                                                    + "_Pipeline_explanation.txt"), "w+")
                    for point in pre_proc_pipeline_points:
                        for key, value in conf.algorithms_explainability_dict.items():
                            if key == point:
                                explanation.write(value + "\n\n")
                    for point in fs_pipeline_points:
                        for key, value in conf.algorithms_explainability_dict.items():
                            if key == point:
                                explanation.write(value + "\n\n")
                    for key, value in conf.algorithms_explainability_dict.items():
                        if key == model:
                            explanation.write(value + "\n\n")
                    hyper_parameter_tuning_model_point = ""
                    if selected_hp_tuning == "GridSearchCV" and len(grid_params) > 8:
                        hyper_parameter_tuning_model_point = "RandomizedSearchCV"
                    elif selected_hp_tuning == "GridSearchCV" and len(grid_params) <= 8:
                        hyper_parameter_tuning_model_point = "GridSearchCV"
                    elif selected_hp_tuning == "RandomizedSearchCV":
                        hyper_parameter_tuning_model_point = "RandomizedSearchCV"
                    for key, value in conf.algorithms_explainability_dict.items():
                        if key == hyper_parameter_tuning_model_point:
                            explanation.write(value + "\n\n")
                    # Columns deleted from the training set
                    explanation.write("Columns that have been removed from the training:" + "\n")
                    for i in columns_to_delete:
                        if i == target_column:
                            explanation.write("This is the target column: " + i + "\n")
                        else:
                            explanation.write("" + i + "\n")

    # #########################################
    # ############ BACK BUTTON ################
    # #########################################

    def back(self):
        from main.main_page import Ui_MainWindow
        main_page = QtWidgets.QMainWindow()
        main_page.ui = Ui_MainWindow()
        main_page.ui.setupUi(main_page)
        main_page.show()

    def main(self):
        app = QtWidgets.QApplication(sys.argv)
        Training = QtWidgets.QDialog()
        ui = Ui_Training()
        ui.setupUi(Training)
        Training.show()
        sys.exit(app.exec_())

        # #########################################
        # ########### RUN PIPELINES ###############
        # #########################################

    def runPipelines(self):
        # self.createIndividual()
        # self.runReasoner()
        import fnmatch
        import subprocess
        import os
        path = self.datasetPathLineEdit.text()
        dataset_name_ext = os.path.basename(path)
        dataset_name = os.path.splitext(dataset_name_ext)[0]
        main_folder = self.newProjectsPathLineEdit.text()
        main_pipeline_path = main_folder
        print(main_pipeline_path)
        pipelines = []
        paths = []
        for directory in os.listdir(main_pipeline_path):
            new_subpath = os.path.join(main_pipeline_path, directory)
            paths.append(os.path.join(main_pipeline_path, directory))
            for file in os.listdir(new_subpath):
                if fnmatch.fnmatch(file, '*pipeline_run.bat'):
                    pipelines.append(os.path.join(new_subpath, file))
        print(pipelines)

        for pipeline in pipelines:
            subprocess.Popen(pipeline)
        self.runPredictionsBtn.setEnabled(True)

    def run_predictions(self):
        prediction = QtWidgets.QDialog()
        prediction.ui = Form1()
        prediction.ui.setupUi(prediction)
        prediction.exec_()
        prediction.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Training = QtWidgets.QDialog()
    ui = Ui_Training()
    ui.setupUi(Training)
    Training.show()
    sys.exit(app.exec_())
