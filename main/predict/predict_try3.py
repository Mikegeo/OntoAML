# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'predict_try2.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLineEdit
from PyQt5.QtGui import QIntValidator
import os
import numpy as np
import pandas as pd
import fnmatch
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
import eli5
import shap
import dill
import joblib
from functools import partial
import webbrowser
from PIL import Image
from fuzzywuzzy import process
from os.path import exists


class Ui_prediction(object):
    def setupUi(self, prediction):
        prediction.setObjectName("Prediction")
        prediction.setEnabled(True)
        prediction.resize(480, 380)
        font = QtGui.QFont()
        font.setKerning(False)
        prediction.setFont(font)
        prediction.setMouseTracking(False)
        self.upload_test_dataset_btn = QtWidgets.QPushButton(prediction)
        self.upload_test_dataset_btn.setGeometry(QtCore.QRect(50, 120, 131, 23))
        self.upload_test_dataset_btn.setAutoDefault(False)
        self.upload_test_dataset_btn.setFlat(False)
        self.upload_test_dataset_btn.setEnabled(False)
        self.upload_test_dataset_btn.setObjectName("upload_test_dataset_btn")
        self.gridLayoutWidget = QtWidgets.QWidget(prediction)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(50, 190, 311, 141))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.show_best_model_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.show_best_model_btn.setAutoDefault(False)
        self.show_best_model_btn.setFlat(False)
        self.show_best_model_btn.setEnabled(False)
        self.show_best_model_btn.setObjectName("show_best_model_btn")
        self.gridLayout.addWidget(self.show_best_model_btn, 1, 0, 1, 1)
        self.show_best_pipeline_explanation_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.show_best_pipeline_explanation_btn.setEnabled(False)
        self.show_best_pipeline_explanation_btn.setAutoDefault(False)
        self.show_best_pipeline_explanation_btn.setFlat(False)
        self.show_best_pipeline_explanation_btn.setObjectName("show_best_pipeline_explanation_btn")
        self.gridLayout.addWidget(self.show_best_pipeline_explanation_btn, 2, 0, 1, 1)
        self.run_eli5_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.run_eli5_btn.setEnabled(False)
        self.run_eli5_btn.setAutoDefault(False)
        self.run_eli5_btn.setFlat(False)
        self.run_eli5_btn.setObjectName("run_eli5_btn")
        self.gridLayout.addWidget(self.run_eli5_btn, 3, 0, 1, 1)
        self.run_LIME_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.run_LIME_btn.setAutoDefault(False)
        self.run_LIME_btn.setFlat(False)
        self.run_LIME_btn.setEnabled(False)
        self.run_LIME_btn.setObjectName("run_LIME_btn")
        self.gridLayout.addWidget(self.run_LIME_btn, 4, 0, 1, 1)
        self.run_SHAP_Values_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.run_SHAP_Values_btn.setAutoDefault(False)
        self.run_SHAP_Values_btn.setFlat(False)
        self.run_SHAP_Values_btn.setEnabled(False)
        self.run_SHAP_Values_btn.setObjectName("run_SHAP_Values_btn")
        self.gridLayout.addWidget(self.run_SHAP_Values_btn, 5, 0, 1, 1)
        # self.metrics_view_text = QtWidgets.QTextBrowser(prediction)
        # self.metrics_view_text.setGeometry(QtCore.QRect(50, 345, 301, 271))
        # self.metrics_view_text.setObjectName("metrics_view_text")
        self.backBtn = QtWidgets.QPushButton(prediction)
        self.backBtn.setGeometry(QtCore.QRect(360, 340, 75, 23))
        self.backBtn.setAutoDefault(False)
        self.backBtn.setObjectName("backBtn")
        self.choose_test_value_label = QtWidgets.QLabel(prediction)
        self.choose_test_value_label.setGeometry(QtCore.QRect(52, 155, 131, 21))
        self.choose_test_value_label.setObjectName("choose_test_value_label")
        self.test_value_number = QtWidgets.QLineEdit(prediction)
        self.test_value_number.setGeometry(QtCore.QRect(190, 155, 51, 23))
        self.test_value_number.setInputMethodHints(QtCore.Qt.ImhDigitsOnly | QtCore.Qt.ImhPreferNumbers)
        self.test_value_number.setValidator(QIntValidator(1, 999999))
        self.test_value_number.setText("")
        self.test_value_number.setClearButtonEnabled(False)
        self.test_value_number.setObjectName("test_value_number")
        self.datasetNameLabel_1 = QtWidgets.QLabel(prediction)
        self.datasetNameLabel_1.setGeometry(QtCore.QRect(190, 122, 271, 21))
        self.datasetNameLabel_1.setFrameShape(QtWidgets.QFrame.Box)
        self.datasetNameLabel_1.setText("")
        self.datasetNameLabel_1.setObjectName("datasetNameLabel_1")
        self.open_eli5_directory_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.open_eli5_directory_btn.setAutoDefault(False)
        self.open_eli5_directory_btn.setEnabled(False)
        self.open_eli5_directory_btn.setObjectName("open_eli5_folder")
        self.gridLayout.addWidget(self.open_eli5_directory_btn, 3, 1, 1, 1)
        self.open_shap_values_directory_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.open_shap_values_directory_btn.setAutoDefault(False)
        self.open_shap_values_directory_btn.setEnabled(False)
        self.open_shap_values_directory_btn.setObjectName("open_shap_values_folder")
        self.gridLayout.addWidget(self.open_shap_values_directory_btn, 5, 1, 1, 1)
        self.open_lime_directory_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.open_lime_directory_btn.setAutoDefault(False)
        self.open_lime_directory_btn.setEnabled(False)
        self.open_lime_directory_btn.setObjectName("open_lime_folder")
        self.gridLayout.addWidget(self.open_lime_directory_btn, 4, 1, 1, 1)
        self.choose_main_pipeline_folder_btn = QtWidgets.QPushButton(prediction)
        self.choose_main_pipeline_folder_btn.setGeometry(QtCore.QRect(50, 80, 141, 21))
        self.choose_main_pipeline_folder_btn.setAutoDefault(False)
        self.choose_main_pipeline_folder_btn.setObjectName("choose_main_pipeline_folder_btn")
        self.main_pipeline_folder_label = QtWidgets.QLabel(prediction)
        self.main_pipeline_folder_label.setGeometry(QtCore.QRect(200, 80, 261, 21))
        self.main_pipeline_folder_label.setFrameShape(QtWidgets.QFrame.Box)
        self.main_pipeline_folder_label.setText("")
        self.main_pipeline_folder_label.setObjectName("main_pipeline_folder_label")
        self.start_again_btn = QtWidgets.QPushButton(prediction)
        self.start_again_btn.setGeometry(QtCore.QRect(390, 155, 71, 23))
        self.start_again_btn.setAutoDefault(False)
        self.start_again_btn.setObjectName("next_prediction_btn")
        self.label = QtWidgets.QLabel(prediction)
        self.label.setGeometry(QtCore.QRect(180, 20, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.retranslateUi(prediction)
        QtCore.QMetaObject.connectSlotsByName(prediction)

        # #############################################
        # ############ connect buttons ################
        # #############################################

        self.choose_main_pipeline_folder_btn.clicked.connect(self.uploadMainPipelineFolder)
        self.upload_test_dataset_btn.clicked.connect(self.uploadTestDataset)
        # self.save_value_btn.clicked.connect(self.saveValueNumber)
        # self.clear_value_btn.clicked.connect(self.clearValueNumber)
        self.start_again_btn.clicked.connect(self.startAgain)
        # show buttons
        # self.show_best_model_btn.clicked.connect(self.bestModel)
        self.show_best_model_btn.clicked.connect(self.showBestHtmlMetrics)
        self.show_best_pipeline_explanation_btn.clicked.connect(self.getBestPipelineExplanation)
        self.run_eli5_btn.clicked.connect(self.eli5)
        self.run_LIME_btn.clicked.connect(self.lime)
        self.run_SHAP_Values_btn.clicked.connect(self.shap_values)
        # open buttons
        self.open_eli5_directory_btn.clicked.connect(self.eli5_directory)
        self.open_lime_directory_btn.clicked.connect(self.lime_directory)
        self.open_shap_values_directory_btn.clicked.connect(self.shap_values_directory)
        # back button
        self.backBtn.clicked.connect(self.back)
        self.backBtn.clicked.connect(prediction.close)

    def retranslateUi(self, Prediction):
        _translate = QtCore.QCoreApplication.translate
        Prediction.setWindowTitle(_translate("Prediction", "Make Predictions"))
        self.upload_test_dataset_btn.setText(_translate("Prediction", "Upload Test Dataset"))
        self.show_best_model_btn.setText(_translate("Prediction", "Show Best Model Metrics"))
        self.show_best_pipeline_explanation_btn.setText(_translate("Prediction", "Show Best Pipeline Explanation"))
        self.run_eli5_btn.setText(_translate("Prediction", "Run Eli5 Explanation"))
        self.run_LIME_btn.setText(_translate("Prediction", "Run LIME Explanation"))
        self.run_SHAP_Values_btn.setText(_translate("Prediction", "Run SHAP Values Explanation"))
        self.backBtn.setText(_translate("Prediction", "Back"))
        self.choose_test_value_label.setText(_translate("Prediction", "Choose Sample Number"))
        self.open_eli5_directory_btn.setText(_translate("Prediction", "Open Eli5 Folder"))
        self.open_shap_values_directory_btn.setText(_translate("Prediction", "Open SHAP Values Folder"))
        self.open_lime_directory_btn.setText(_translate("Prediction", "Open LIME Folder"))
        self.choose_main_pipeline_folder_btn.setText(_translate("Prediction", "Choose Experiment Folder"))
        self.start_again_btn.setText(_translate("Prediction", "Start Again"))
        self.label.setText(_translate("Prediction", "Make Predictions"))

    # ############################################
    # ########## UPLOAD MAIN PIPELINE ############
    # ############################################

    def uploadMainPipelineFolder(self):
        self.main_pipeline_folder_label.clear()
        dialog = QFileDialog()
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(os.getenv('HOME'))
        self.main_pipeline_folder_label.setText(folder_path)
        if self.main_pipeline_folder_label.text() != "":
            # upload dataset button
            self.upload_test_dataset_btn.setEnabled(True)
            # show buttons
            self.show_best_model_btn.setEnabled(True)
            self.show_best_pipeline_explanation_btn.setEnabled(True)
            self.run_eli5_btn.setEnabled(True)
            self.run_LIME_btn.setEnabled(True)
            self.run_SHAP_Values_btn.setEnabled(True)
            # show directory buttons
            self.open_eli5_directory_btn.setEnabled(True)
            self.open_lime_directory_btn.setEnabled(True)
            self.open_shap_values_directory_btn.setEnabled(True)
        view_folder_name = folder_path.split("/")[-1]

    # ############################################
    # ############# UPLOAD DATASET ###############
    # ############################################

    def uploadTestDataset(self):
        self.datasetNameLabel_1.clear()
        dialog = QFileDialog()
        filepath = dialog.getOpenFileName(os.getenv('HOME'))
        path = filepath[0]
        self.datasetNameLabel_1.setText(path)
        # if self.datasetNameLabel_1.text() != "":
        #     self.save_value_btn.setEnabled(True)
        #     self.clear_value_btn.setEnabled(True)
        import sklearn
        print(sklearn.__version__)
    # ############################################
    # ############## SAVE VALUE ##################
    # ############################################

    # def saveValueNumber(self):
    #     test_value_number = QLineEdit()
    #     test_value = self.test_value_number.text()
    #     if not test_value:
    #         exception = QMessageBox()
    #         exception.setWindowTitle("Error")
    #         exception.setIcon(QMessageBox.Information)
    #         exception.setText('Please add a value number.')
    #         x = exception.exec_()
    #     try:
    #         number = int(test_value)
    #         if type(number) == int:
    #             self.test_value_number.setDisabled(True)
    #             self.run_eli5_btn.setEnabled(True)
    #             self.run_LIME_btn.setEnabled(True)
    #             self.run_SHAP_Values_btn.setEnabled(True)
    #     except Exception:
    #         exception = QMessageBox()
    #         exception.setWindowTitle("Error")
    #         exception.setIcon(QMessageBox.Information)
    #         exception.setText('Error Input can only be a number')
    #         x = exception.exec_()

    # ########################################
    # ######### CLEAR FUNCTIONS ##############
    # ########################################

    # def clearValueNumber(self):
    #     test_value_number = QLineEdit()
    #     self.test_value_number.setDisabled(False)
    #     self.test_value_number.clear()
    #     self.run_eli5_btn.setEnabled(False)
    #     self.run_LIME_btn.setEnabled(False)
    #     self.run_SHAP_Values_btn.setEnabled(False)

    def startAgain(self):
        test_value_number = QLineEdit()
        self.test_value_number.setDisabled(False)
        self.test_value_number.clear()
        self.main_pipeline_folder_label.clear()
        self.datasetNameLabel_1.clear()
        # self.metrics_view_text.clear()
        # disable dataset button
        self.upload_test_dataset_btn.setEnabled(False)
        # disable value buttons
        # self.save_value_btn.setEnabled(False)
        # self.clear_value_btn.setEnabled(False)
        # disable show results buttons
        self.show_best_model_btn.setEnabled(False)
        self.show_best_pipeline_explanation_btn.setEnabled(False)
        self.run_eli5_btn.setEnabled(False)
        self.run_LIME_btn.setEnabled(False)
        self.run_SHAP_Values_btn.setEnabled(False)
        # disable open directory buttons
        self.open_eli5_directory_btn.setEnabled(False)
        self.open_lime_directory_btn.setEnabled(False)
        self.open_shap_values_directory_btn.setEnabled(False)

    # #################################################
    # ########## BEST MODEL METRICS ###################
    # #################################################

    def bestModel(self):
        main_path = self.main_pipeline_folder_label.text()
        # Get the paths of the trained pipelines (.joblib) and the paths of the trained pipeline scores (.txt)
        # and create 2 arrays
        trained_pipelines_paths = []
        trained_pipelines_scores_paths = []
        trained_pipelines_directory_paths = []
        for directory in os.listdir(main_path):
            new_path = os.path.join(main_path, directory)
            trained_pipelines_directory_paths.append(new_path)
            for file in os.listdir(new_path):
                if fnmatch.fnmatch(file, '*pipeline.joblib'):
                    trained_pipelines_paths.append(os.path.join(new_path, file))
                if fnmatch.fnmatch(file, '*scores.txt'):
                    trained_pipelines_scores_paths.append(os.path.join(new_path, file))
        # Get the trained pipelines (.joblib) and print the to check their algorithms
        for pipeline_path in trained_pipelines_paths:
            print(pipeline_path)
            pipeline = joblib.load(pipeline_path)
        # Get the model's names and the accuracies of the trained pipelines from their metrics txt file
        # and create 2 arrays
        trained_pipelines_model_name = []
        trained_pipelines_accuracies = []
        trained_pipelines_f1_scores = []
        trained_pipelines_auc_scores = []
        trained_pipelines_r2_scores = []
        problems_type = []
        for trained_pipeline_score in trained_pipelines_scores_paths:
            f = open(trained_pipeline_score, 'r')
            for line in f.readlines():
                line = line.strip()
                if "Problem: " in line:
                    problem = line.split(": ")[1]
                    problems_type.append(problem)
                if "model's name:" in line:
                    models_name = line.split(":")[1]
                    trained_pipelines_model_name.append(models_name)
                if "Accuracy score:" in line:
                    accuracy_score = line.split(":")[1]
                    trained_pipelines_accuracies.append(accuracy_score)
                if "F1 score" in line:
                    f1_score = line.split(":")[1]
                    trained_pipelines_f1_scores.append(f1_score)
                if "Auc score" in line:
                    auc_score = line.split(":")[1]
                    trained_pipelines_auc_scores.append(auc_score)
                if "r2:" in line:
                    r2_score = line.split(":")[1]
                    trained_pipelines_r2_scores.append(r2_score)
        # Get the best model
        best_name_pipeline = []
        # create a dict of models with f1 scores
        if len(best_name_pipeline) == 0:
            if len(trained_pipelines_f1_scores) != 0:
                trained_pipelines_names_and_f1_scores = dict(
                    zip(trained_pipelines_model_name, trained_pipelines_f1_scores))
                all_f1_scores = trained_pipelines_names_and_f1_scores.values()
                max_f1_score = max(all_f1_scores)
                best_name_pipeline_list_f1 = []
                for key, value in trained_pipelines_names_and_f1_scores.items():
                    if value == max_f1_score:
                        best_name_pipeline_list_f1.append(key)
                if len(best_name_pipeline_list_f1) == 1:
                    best_name_pipeline.append(best_name_pipeline_list_f1[0])
                else:
                    pass
        if len(best_name_pipeline) == 0:
            if len(trained_pipelines_auc_scores) != 0:
                # create a dict of models with auc scores
                trained_pipelines_names_and_auc_scores = dict(
                    zip(trained_pipelines_model_name, trained_pipelines_auc_scores))
                all_auc_scores = trained_pipelines_names_and_auc_scores.values()
                max_auc_score = max(all_auc_scores)
                best_name_pipeline_list_auc = []
                for key, value in trained_pipelines_names_and_auc_scores.items():
                    if value == max_auc_score:
                        best_name_pipeline_list_auc.append(key)
                if len(best_name_pipeline_list_auc) == 1:
                    best_name_pipeline.append(best_name_pipeline_list_auc[0])
                else:
                    pass
        if len(best_name_pipeline) == 0:
            # Create a dictionary of the models and their accuracies and get the best model with acc
            if len(trained_pipelines_accuracies) != 0:
                trained_pipelines_names_and_accuracy_score = dict(
                    zip(trained_pipelines_model_name, trained_pipelines_accuracies))
                all_accuracies_scores = trained_pipelines_names_and_accuracy_score.values()
                max_accuracy = max(all_accuracies_scores)
                best_name_pipeline_list_accu = []
                for key, value in trained_pipelines_names_and_accuracy_score.items():
                    if value == max_accuracy:
                        best_name_pipeline_list_accu.append(key)
                if len(best_name_pipeline_list_accu) == 1:
                    best_name_pipeline.append(best_name_pipeline_list_accu[0])
                else:
                    pass
        if len(best_name_pipeline) == 0:
            # create a dict of models with f1 scores
            if len(trained_pipelines_f1_scores) != 0:
                trained_pipelines_names_and_f1_scores = dict(
                    zip(trained_pipelines_model_name, trained_pipelines_f1_scores))
                all_f1_scores = trained_pipelines_names_and_f1_scores.values()
                max_f1_score = max(all_f1_scores)
                best_name_pipeline_list_f1_2 = []
                for key, value in trained_pipelines_names_and_f1_scores.items():
                    if value == max_f1_score:
                        best_name_pipeline_list_f1_2.append(key)
                if len(best_name_pipeline_list_f1_2) == 1:
                    best_name_pipeline.append(best_name_pipeline_list_f1_2[0])
                else:
                    best_name_pipeline.append(best_name_pipeline_list_f1_2[0])
        # #################################
        # ### REGRESSION MODELS ###########
        # #################################
        # create a dict of models with r2 scores
        if len(trained_pipelines_r2_scores) != 0:
            trained_pipelines_names_and_r2_scores = dict(
                zip(trained_pipelines_model_name, trained_pipelines_r2_scores))
            all_r2_scores = trained_pipelines_names_and_r2_scores.values()
            max_r2_score = max(all_r2_scores)
            best_name_pipeline_list_r2 = []
            for key, value in trained_pipelines_names_and_r2_scores.items():
                if value == max_r2_score:
                    best_name_pipeline_list_r2.append(key)
            if len(best_name_pipeline_list_r2) == 1:
                best_name_pipeline.append(best_name_pipeline_list_r2[0])
            else:
                best_name_pipeline.append(best_name_pipeline_list_r2[0])
        else:
            pass
        best_directory_path_list = []
        for pipeline_name in best_name_pipeline:
            best_directory_path_list.append(process.extractOne(pipeline_name, trained_pipelines_directory_paths)[0])
        best_directory_path = (', '.join(best_directory_path_list))
        scores_file_lines = []
        for file in os.listdir(best_directory_path):
            if fnmatch.fnmatch(file, '*scores.txt'):
                scores_file = os.path.join(best_directory_path, file)
                f = open(scores_file, 'r')
                for line in f.readlines():
                    scores_file_lines.append(line)
        task = problems_type[0]
        return best_directory_path, task

    # ##################################
    # #### SHOW HTML METRICS ###########
    # ##################################

    def showBestHtmlMetrics(self):
        pipes_path, task = self.bestModel()
        for file in os.listdir(pipes_path):
            if fnmatch.fnmatch(file, '*scores.html'):
                html_scores_file = os.path.join(pipes_path, file)
                webbrowser.open(html_scores_file, new=2)

    # ###################################
    # #### GET PIPELINE EXPLANATION #####
    # ###################################

    def getBestPipelineExplanation(self):
        pipes_path, task = self.bestModel()
        explanation_file_lines = []
        for file in os.listdir(pipes_path):
            if fnmatch.fnmatch(file, '*explanation.txt'):
                best_pipeline_explanation_file = os.path.join(pipes_path, file)
                f = open(best_pipeline_explanation_file, 'r')
                for line in f.readlines():
                    explanation_file_lines.append(line)
        for file in os.listdir(pipes_path):
            if fnmatch.fnmatch(file, '*pipeline_diagram.html'):
                best_pipeline_diagrame_file = os.path.join(pipes_path, file)
                if best_pipeline_diagrame_file:
                    f = open(best_pipeline_diagrame_file, 'r')
                    if "Explanation" in f.read():
                        webbrowser.open(best_pipeline_diagrame_file, new=2)
                        return
                    else:
                        f = open(best_pipeline_diagrame_file, 'a', encoding='utf-8')
                        f.write("<p> " + "Explanation" + "<br><br>" + "\n")
                        f.write("<br> ".join(map(str, explanation_file_lines)))
                        webbrowser.open(best_pipeline_diagrame_file, new=2)

    # ###################################
    # #### GET EXPLAINABLE FUNCTION #####
    # ###################################
    def check_new_selected_features(self):
        pipes_path, task = self.bestModel()
        main_path = self.main_pipeline_folder_label.text()
        selected_features_file = os.path.join(pipes_path, "selected_features")
        selected_features_exist = os.path.isfile(selected_features_file)
        check_new_selected_features = []
        if selected_features_exist:
            for filename in os.listdir(pipes_path):
                # load features selected from the feature selection algorithm
                if fnmatch.fnmatch(filename, 'selected_features'):
                    with open(os.path.join(pipes_path, filename), "rb") as f:
                        selected_features = dill.load(f)
                        # remove the values from the selected features
                        for feature in selected_features:
                            check_new_selected_features.append(feature.split("__", 1)[0])
        else:
            check_new_selected_features.append("no_selected_features")
        return check_new_selected_features


    def getExplainableFile(self):
        pipes_path, task = self.bestModel()
        main_path = self.main_pipeline_folder_label.text()
        new_all_features = []
        new_selected_features = []
        # load needed files
        for filename in os.listdir(pipes_path):
            # load pipeline
            if fnmatch.fnmatch(filename, '*pipeline.joblib'):
                with open(os.path.join(pipes_path, filename), 'rb') as f:
                    pipeline = joblib.load(f)
            # load categorical names dictionary
            if fnmatch.fnmatch(filename, 'categorical_names.pkl'):
                with open(os.path.join(pipes_path, filename), 'rb') as f:
                    categorical_names = dill.load(f)
            # load full categorical names dictionary
            if fnmatch.fnmatch(filename, 'full_categorical_names.pkl'):
                with open(os.path.join(pipes_path, filename), 'rb') as f:
                    full_categorical_names = dill.load(f)
            # load explainer
            if fnmatch.fnmatch(filename, 'lime_explainer_file'):
                with open(os.path.join(pipes_path, filename), 'rb') as f:
                    lime_explainer = dill.load(f)
            # load preprocessor
            if fnmatch.fnmatch(filename, 'Data_preprocessor.joblib'):
                with open(os.path.join(pipes_path, filename), "rb") as f:
                    data_prepro = joblib.load(f)
            # load shap explainer
            if fnmatch.fnmatch(filename, '*shap_explainer'):
                with open(os.path.join(pipes_path, filename), "rb") as f:
                    shap_explainer_path = str(os.path.join(pipes_path, filename))
                    shap_explainer_name = shap_explainer_path.split("\\")[-1]
                    shap_explainer = dill.load(f)
            # load all features
            if fnmatch.fnmatch(filename, 'all_features'):
                with open(os.path.join(pipes_path, filename), "rb") as f:
                    all_features = dill.load(f)
                    # remove the values and duplicate features from one hot from all features
                    for feature in all_features:
                        new_all_features.append(feature.split("__", 1)[0])
                        new_all_features = list(dict.fromkeys(new_all_features))
        # load features selected from the feature selection algorithm
        selected_features_file = os.path.join(pipes_path, "selected_features")
        selected_features_exist = os.path.isfile(selected_features_file)
        if selected_features_exist:
            with open(os.path.join(selected_features_file), "rb") as f:
                selected_features = dill.load(f)
                new_selected_features = self.check_new_selected_features()
        else:
            new_selected_features = self.check_new_selected_features()
        if new_selected_features[0] != "no_selected_features":
            return pipeline, categorical_names, full_categorical_names, lime_explainer, data_prepro, \
                   shap_explainer, all_features, new_all_features, selected_features, shap_explainer_name, \
                   new_selected_features
        else:
            return pipeline, categorical_names, full_categorical_names, lime_explainer, data_prepro,\
                   shap_explainer, all_features, new_all_features, shap_explainer_name

    # #################################################
    # ########## ELI5 MODEL WEIGHTS ###################
    # #################################################

    def eli5_directory(self):
        pipes_path, task = self.bestModel()
        explainable_folder_main = "main_explainable_folder"
        eli5_folder = "eli5_folder"
        eli5_folder_path = os.path.join(pipes_path, explainable_folder_main, eli5_folder)
        webbrowser.open(eli5_folder_path)

    def eli5(self):
        pipes_path, task = self.bestModel()
        target_columns_file = os.path.join(pipes_path, "target_labels.pkl")
        target_columns_exist = os.path.isfile(target_columns_file)
        if target_columns_exist:
            with open(os.path.join(target_columns_file), "rb") as f:
                target_columns = dill.load(f)
        if self.test_value_number.text() != "" and self.datasetNameLabel_1.text() != "":
            i = int(self.test_value_number.text())
            pipes_path, task = self.bestModel()
            explainable_folder_main = "main_explainable_folder"
            eli5_folder = "eli5_folder"
            eli5_folder_path = os.path.join(pipes_path, explainable_folder_main, eli5_folder)
            # perm importance file
            perm_impo_file = os.path.join(eli5_folder_path, 'permutation_importance.html')
            perm_impo_file_exist = os.path.isfile(perm_impo_file)
            path = self.datasetNameLabel_1.text()
            with open(path, "r") as file:
                missing_values_formats = ['n/a', 'na', '--', '?', ' ', 'NA', 'N/A']
                reader = pd.read_csv(file, header=0, na_values=missing_values_formats, sep=None, engine='python',
                                     encoding='UTF-8')
                df = pd.DataFrame(data=reader)
                df_size = len(df)
                real_df_size = str(df_size-1)
                if i > int(real_df_size):
                    exception = QMessageBox()
                    exception.setWindowTitle("Error")
                    exception.setIcon(QMessageBox.Information)
                    exception.setText("The dataset value size is " +
                                      "" + real_df_size + " please change your test sample number.")
                    x = exception.exec_()
                    return None
            # check if there is feature selection file
            check_new_selected_features = self.check_new_selected_features()
            if check_new_selected_features[0] != "no_selected_features":
                pipeline, categorical_names, full_categorical_names, lime_explainer, data_prepro, shap_explainer, \
                all_features, new_all_features, selected_features, shap_explainer_name, new_selected_features = \
                    self.getExplainableFile()
            else:
                pipeline, categorical_names, full_categorical_names, lime_explainer, data_prepro, shap_explainer,\
                all_features, new_all_features, shap_explainer_name = self.getExplainableFile()

            def convert_to_lime_format(x, categorical_names, col_names=None, invert=False):
                if not isinstance(x, pd.DataFrame):
                    x_lime = pd.DataFrame(x, columns=col_names)
                else:
                    x_lime = x.copy()
                for k, v in categorical_names.items():
                    if not invert:
                        label_map = {
                            str_label: int_label for int_label, str_label in enumerate(v)}
                    else:
                        label_map = {
                            int_label: str_label for int_label, str_label in enumerate(v)}
                    x_lime.iloc[:, k] = x_lime.iloc[:, k].map(label_map)
                return x_lime

            # def shap_and_eli5_custom_format(x):
            #     j_data = convert_to_lime_format(x, categorical_names)
            #     try:
            #         imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
            #         imp_mean.fit(j_data)
            #     except:
            #         imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            #         imp_mean.fit(j_data)
            #     custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
            #     print(custom_data)
            #     print(x.columns)
            #     if check_new_selected_features[0] != "no_selected_features":
            #         return custom_data[new_selected_features]
            #     else:
            #         data = x
            #         ohe_data = pd.DataFrame(data_prepro.transform(data), columns=all_features)
            #         return ohe_data
            def shap_and_eli5_custom_format(x):
                j_data = convert_to_lime_format(x, categorical_names)
                print(j_data)
                mean_imputer = SimpleImputer(strategy='mean')
                frequent_imputer = SimpleImputer(strategy='most_frequent')

                custom_data = pd.DataFrame(columns=x.columns, index=j_data.index)
                for col in j_data.columns:
                    if j_data[col].dtype == 'float' or j_data[col].dtype == 'int':
                        mean_imputer.fit(j_data[[col]])
                        custom_data.loc[:, [col]] = mean_imputer.transform(j_data[[col]])
                    else:
                        frequent_imputer.fit(j_data[[col]])
                        custom_data.loc[:, [col]] = frequent_imputer.transform(j_data[[col]])
                print(custom_data)
                if check_new_selected_features[0] != "no_selected_features":
                    return custom_data[new_selected_features]
                else:
                    data = x
                    ohe_data = pd.DataFrame(data_prepro.transform(data), columns=all_features)
                    return ohe_data

            # def imputed_data(x):
            #     if df.isna().any().any():
            #         data = df
            #         imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
            #         imp_mean.fit(data)
            #         shap_imputed_data = pd.DataFrame(imp_mean.fit_transform(data), columns=x.columns, index=data.index)
            #         return shap_imputed_data
            #     else:
            #         return df
            #
            # shap_and_eli5_dataset1 = data_prepro.transform(df)
            # shap_and_eli5_dataset = shap_and_eli5_custom_format(df)
            # pipeline's model
            pipeline_model = pipeline.named_steps["" + pipeline.steps[-1][0] + ""]
           # print(pipeline_model.predict(shap_and_eli5_dataset))
            with open(path, "r") as file:
                missing_values_formats = ['n/a', 'na', '--', '?', ' ', 'NA', 'N/A']
                reader = pd.read_csv(file, header=0, na_values=missing_values_formats, sep=None, engine='python',
                                     encoding='UTF-8')
                df = pd.DataFrame(data=reader)
                # convert the test dataframe to Eli5 format
                if len(categorical_names) != 0:
                    shap_and_eli5_dataset = shap_and_eli5_custom_format(df)
                elif len(full_categorical_names) != 0:
                    shap_and_eli5_dataset = shap_and_eli5_custom_format(df)
                else:
                    shap_and_eli5_dataset = df
            # shap_and_eli5_dataset1 = data_prepro.transform(df)
            #     shap_and_eli5_dataset = pd.DataFrame(shap_and_eli5_dataset1, columns=all_features)
            #     print(shap_and_eli5_dataset.to_string)
                # show the local prediction of a single selected value
                eli5_observation = shap_and_eli5_dataset.iloc[[i], :]
                eli5_observation_index = shap_and_eli5_dataset.index[i]
                # show Eli5 graphs depending on if the pipeline has feature selection or not
                if check_new_selected_features[0] != "no_selected_features":
                    pipeline, categorical_names, full_categorical_names, lime_explainer, data_prepro, shap_explainer, \
                    all_features, new_all_features, selected_features, shap_explainer_name, new_selected_features = \
                        self.getExplainableFile()
                    # show weights of the model in the pipeline if the pipeline has feature selection
                    # eli5_weights = eli5.show_weights(pipeline_model, feature_names=selected_features)
                    # with open(os.path.join(eli5_folder_path, 'eli5_weights.html'), 'wb') as f:
                    #     f.write(eli5_weights.data.encode("UTF-8"))
                    #     eli5_weights_file = os.path.join(eli5_folder_path, 'eli5_weights.html')
                    # show local prediction of the selected row (pipeline with feature selection)
                    from eli5.sklearn import PermutationImportance
                    #print(pipeline_model.predict(df))
                    if target_columns_exist:
                        with open(os.path.join(target_columns_file), "rb") as f:
                            target_columns = dill.load(f)
                            print(target_columns)
                        eli5_weights = eli5.show_weights(pipeline_model, target_names=target_columns,
                                                         feature_names=selected_features)
                        eli5_predictions = eli5.show_prediction(pipeline_model, eli5_observation,
                                                                feature_names=selected_features,
                                                                target_names=target_columns)
                    else:
                        eli5_weights = eli5.show_weights(pipeline_model, feature_names=selected_features)
                        eli5_predictions = eli5.show_prediction(pipeline_model, eli5_observation)
                    with open(os.path.join(eli5_folder_path, 'eli5_weights.html'), 'wb') as f:
                        f.write(eli5_weights.data.encode("UTF-8"))
                        eli5_weights_file = os.path.join(eli5_folder_path, 'eli5_weights.html')
                    with open(os.path.join(eli5_folder_path, '' + str(eli5_observation_index) + '_eli5_explanation.html'),
                              'wb') as f:
                        f.write(eli5_predictions.data.encode("UTF-8"))
                        eli5_local_prediction_file = os.path.join(eli5_folder_path, '' + str(eli5_observation_index)
                                                                  + '_eli5_explanation.html')
                        #print(pipeline_model.predict(eli5_observation))
                        #print(pipeline_model.predict_proba(eli5_observation))
                        # Message Box open prediction
                        if os.path.isfile(eli5_weights_file):
                            msg_done = QMessageBox()
                            msg_done.setWindowTitle("Done")
                            msg_done.setIcon(QMessageBox.Question)
                            msg_done.setText("Prediction Done do you want to open the prediction?")
                            msg_done.setStandardButtons(QMessageBox.Open | QMessageBox.Cancel | QMessageBox.Yes |
                                                        QMessageBox.No)
                            open_btn = msg_done.button(QMessageBox.Open)
                            open_btn.setText("Show Test Sample Local Prediction")
                            weights_btn = msg_done.button(QMessageBox.Yes)
                            weights_btn.setText("Show Eli5 Feature Importance")
                            if perm_impo_file_exist:
                                perm_imp_btn = msg_done.button(QMessageBox.No)
                                perm_imp_btn.setText("Show Permutation Importance")
                            x = msg_done.exec_()
                            if msg_done.clickedButton() == open_btn:
                                webbrowser.open(eli5_local_prediction_file, new=2)
                            if msg_done.clickedButton() == weights_btn:
                                webbrowser.open(eli5_weights_file, new=2)
                            if msg_done.clickedButton() == perm_imp_btn:
                                webbrowser.open(perm_impo_file, new=2)
                else:
                    # show weights of the model in the pipeline if the pipeline does not have feature selection
                    # eli5_weights = eli5.show_weights(pipeline_model,
                    #                                  feature_names=all_features)
                    # with open(os.path.join(eli5_folder_path, 'eli5_weights.html'), 'wb') as f:
                    #     f.write(eli5_weights.data.encode("UTF-8"))
                    #     eli5_weights_file = os.path.join(eli5_folder_path, 'eli5_weights.html')
                    #     print(eli5_weights_file)
                    # show local prediction of the selected row (pipeline with no feature selection)
                    if target_columns_exist:
                        with open(os.path.join(target_columns_file), "rb") as f:
                            target_columns = dill.load(f)
                        eli5_weights = eli5.show_weights(pipeline_model, target_names=target_columns,
                                                         feature_names=all_features)
                        eli5_predictions = eli5.show_prediction(pipeline_model, eli5_observation,
                                                                target_names=target_columns, feature_names=all_features)
                    else:
                        eli5_weights = eli5.show_weights(pipeline_model,
                                                         feature_names=all_features)
                        eli5_predictions = eli5.show_prediction(pipeline_model, eli5_observation,
                                                                feature_names=all_features)
                    with open(os.path.join(eli5_folder_path, 'eli5_weights.html'), 'wb') as f:
                        f.write(eli5_weights.data.encode("UTF-8"))
                        eli5_weights_file = os.path.join(eli5_folder_path, 'eli5_weights.html')
                    with open(os.path.join(eli5_folder_path, '' + str(eli5_observation_index) + '_eli5_explanation.html'),
                              'wb') as f:
                        f.write(eli5_predictions.data.encode("UTF-8"))
                        eli5_local_prediction_file = os.path.join(eli5_folder_path, '' + str(eli5_observation_index)
                                                                  + '_eli5_explanation.html')
                        # Message Box open prediction
                        if os.path.isfile(eli5_weights_file):
                            msg_done = QMessageBox()
                            msg_done.setWindowTitle("Prediction Finished!")
                            msg_done.setIcon(QMessageBox.Question)
                            msg_done.setText("Prediction done do you want to open the Eli5 prediction?")
                            msg_done.setStandardButtons(QMessageBox.Open | QMessageBox.Cancel | QMessageBox.Yes
                                                        | QMessageBox.No)
                            open_btn = msg_done.button(QMessageBox.Open)
                            open_btn.setText("Show Test Sample Local Prediction")
                            weights_btn = msg_done.button(QMessageBox.Yes)
                            weights_btn.setText("Show Eli5 Feature Importance")
                            if perm_impo_file_exist:
                                perm_imp_btn = msg_done.button(QMessageBox.No)
                                perm_imp_btn.setText("Show Permutation Importance")
                            x = msg_done.exec_()
                            if msg_done.clickedButton() == open_btn:
                                webbrowser.open(eli5_local_prediction_file, new=2)
                            if msg_done.clickedButton() == weights_btn:
                                webbrowser.open(eli5_weights_file, new=2)
                            if msg_done.clickedButton() == perm_imp_btn:
                                webbrowser.open(perm_impo_file, new=2)
                    return
        else:
            exception = QMessageBox()
            exception.setWindowTitle("Error")
            exception.setIcon(QMessageBox.Information)
            exception.setText('Error: Please add the Test dataset or the Test Sample Number.')
            x = exception.exec_()
            return None

    # #########################################
    # ########### LIME VALUES #################
    # #########################################

    def lime_directory(self):
        pipes_path, task = self.bestModel()
        explainable_folder_main = "main_explainable_folder"
        lime_folder = "lime_folder"
        lime_folder_path = os.path.join(pipes_path, explainable_folder_main, lime_folder)
        webbrowser.open(lime_folder_path)

    def lime(self):
        if self.test_value_number.text() != "" and self.datasetNameLabel_1.text() != "":
            pipes_path, task = self.bestModel()
            explainable_folder_main = "main_explainable_folder"
            lime_folder = "lime_folder"
            lime_folder_path = os.path.join(pipes_path, explainable_folder_main, lime_folder)
            path = self.datasetNameLabel_1.text()
            # value for local explanation from the imputed lime format dataframe
            i = int(self.test_value_number.text())
            with open(path, "r") as file:
                missing_values_formats = ['n/a', 'na', '--', '?', ' ', 'NA', 'N/A']
                reader = pd.read_csv(file, header=0, na_values=missing_values_formats, sep=None, engine='python',
                                     encoding='UTF-8')
                df = pd.DataFrame(data=reader)
                df_size = len(df)
                real_df_size = str(df_size-1)
                if i > int(real_df_size):
                    exception = QMessageBox()
                    exception.setWindowTitle("Error")
                    exception.setIcon(QMessageBox.Information)
                    exception.setText("The dataset value size is " +
                                      "" + real_df_size + " please change your test value.")
                    x = exception.exec_()
                    return None
            # check if there is feature selection file
            check_new_selected_features = self.check_new_selected_features()
            if check_new_selected_features[0] != "no_selected_features":
                pipeline, categorical_names, full_categorical_names, lime_explainer, data_prepro, shap_explainer, \
                all_features, new_all_features, selected_features, shap_explainer_name, new_selected_features = \
                    self.getExplainableFile()
            else:
                pipeline, categorical_names, full_categorical_names, lime_explainer, data_prepro, shap_explainer, \
                all_features, new_all_features, shap_explainer_name = self.getExplainableFile()

            def convert_to_lime_format(x, categorical_names, col_names=None, invert=False):
                if not isinstance(x, pd.DataFrame):
                    x_lime = pd.DataFrame(x, columns=col_names)
                else:
                    x_lime = x.copy()
                for k, v in categorical_names.items():
                    if not invert:
                        label_map = {
                            str_label: int_label for int_label, str_label in enumerate(v)}
                    else:
                        label_map = {
                            int_label: str_label for int_label, str_label in enumerate(v)}
                    x_lime.iloc[:, k] = x_lime.iloc[:, k].map(label_map)
                return x_lime

            def convert_to_imputed_lime_format(x):
                if len(categorical_names) != 0:
                    lime_data = convert_to_lime_format(x, categorical_names)
                elif len(full_categorical_names) != 0:
                    lime_data = convert_to_lime_format(x, categorical_names)
                else:
                    lime_data = df
                try:
                    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
                    imp_mean.fit(lime_data)
                except:
                    imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                    imp_mean.fit(lime_data)
                imputed_lime_data = pd.DataFrame(imp_mean.fit_transform(lime_data), columns=x.columns,
                                                 index=lime_data.index)
                return imputed_lime_data

            with open(path, "r") as file:
                missing_values_formats = ['n/a', 'na', '--', '?', ' ', 'NA', 'N/A']
                reader = pd.read_csv(file, header=0, na_values=missing_values_formats, sep=None, engine='python',
                                     encoding='UTF-8')
                df = pd.DataFrame(data=reader)

            # convert the test dataframe to lime format
            lime_format_df = convert_to_imputed_lime_format(df)
            X_observation = lime_format_df.iloc[[i], :]
            X_observation1 = lime_format_df.values[i]
            # index is the row number in the dataset of the value
            X_index = lime_format_df.index[i]
            pipeline_model = pipeline.named_steps["" + pipeline.steps[-1][0] + ""]
            if len(categorical_names) != 0:
                def custom_predict_proba(x, model):
                    X_str = convert_to_lime_format(x, categorical_names, col_names=lime_format_df.columns, invert=True)
                    if task == "Classification":
                        return model.predict_proba(X_str)
                    else:
                        return model.predict(X_str)

                predict_proba = partial(custom_predict_proba, model=pipeline)
                # Lime explanation of the selected value
                explanation = lime_explainer.explain_instance(X_observation.values[0], predict_proba, num_features=15)

            elif len(full_categorical_names) != 0:
                def custom_predict_proba(x, model):
                    X_str = convert_to_lime_format(x, categorical_names, col_names=lime_format_df.columns, invert=True)
                    if task == "Classification":
                        return model.predict_proba(X_str)
                    else:
                        return model.predict(X_str)

                predict_proba = partial(custom_predict_proba, model=pipeline)
                # Lime explanation of the selected value
                explanation = lime_explainer.explain_instance(X_observation.values[0], predict_proba, num_features=15)
            else:
                if task == "Classification":
                    predict_proba = pipeline.named_steps["" + pipeline.steps[-1][0] + ""].predict_proba
                else:
                    predict_proba = pipeline.named_steps["" + pipeline.steps[-1][0] + ""].predict
                # Lime explanation of the selected value
                explanation = lime_explainer.explain_instance(X_observation.values[0], predict_proba, num_features=15)

            # Lime explanation of the selected value
            # export the local explanation in html format
            explanation.save_to_file(os.path.join(lime_folder_path, "" + str(X_index) + "_row_lime_explanation.html"))
            lime_prediction_file = os.path.join(lime_folder_path, "" + str(X_index) + "_row_lime_explanation.html")
            # Message Box open prediction
            if os.path.isfile(lime_prediction_file):
                msg_done = QMessageBox()
                msg_done.setWindowTitle("Prediction Finished!")
                msg_done.setIcon(QMessageBox.Question)
                msg_done.setText("Prediction done do you want to open the LIME prediction?")
                msg_done.setStandardButtons(QMessageBox.Open | QMessageBox.Cancel)
                open_btn = msg_done.button(QMessageBox.Open)
                x = msg_done.exec_()
                if msg_done.clickedButton() == open_btn:
                    webbrowser.open(lime_prediction_file, new=2)
            return
        else:
            exception = QMessageBox()
            exception.setWindowTitle("Error")
            exception.setIcon(QMessageBox.Information)
            exception.setText('Error: Please add the Test dataset or the Sample Number.')
            x = exception.exec_()
            return None

    # ###################################
    # #### SHAP-VALUES ##################
    # ###################################

    def shap_values_directory(self):
        pipes_path, task = self.bestModel()
        explainable_folder_main = "main_explainable_folder"
        shap_values_folder = "shap_values_folder"
        shap_values_folder_path = os.path.join(pipes_path, explainable_folder_main, shap_values_folder)
        webbrowser.open(shap_values_folder_path)

    def shap_values(self):
        if self.test_value_number.text() != "" and self.datasetNameLabel_1.text() != "":
            pipes_path, task = self.bestModel()
            explainable_folder_main = "main_explainable_folder"
            shap_values_folder = "shap_values_folder"
            shap_values_folder_path = os.path.join(pipes_path, explainable_folder_main, shap_values_folder)
            path = self.datasetNameLabel_1.text()
            # value for local explanation from the imputed lime format dataframe
            i = int(self.test_value_number.text())
            with open(path, "r") as file:
                missing_values_formats = ['n/a', 'na', '--', '?', ' ', 'NA', 'N/A']
                reader = pd.read_csv(file, header=0, na_values=missing_values_formats, sep=None, engine='python',
                                     encoding='UTF-8')
                df = pd.DataFrame(data=reader)
                df_size = len(df)
                real_df_size = str(df_size-1)
                if i > int(real_df_size):
                    exception = QMessageBox()
                    exception.setWindowTitle("Error")
                    exception.setIcon(QMessageBox.Information)
                    exception.setText("The dataset value size is " +
                                      "" + real_df_size + " please change your test value.")
                    x = exception.exec_()
                    return None
            # check if there is feature selection file
            check_new_selected_features = self.check_new_selected_features()
            if check_new_selected_features[0] != "no_selected_features":
                pipeline, categorical_names, full_categorical_names, lime_explainer, data_prepro, shap_explainer, \
                all_features, new_all_features, selected_features, shap_explainer_name, new_selected_features = \
                    self.getExplainableFile()
            else:
                pipeline, categorical_names, full_categorical_names, lime_explainer, data_prepro, shap_explainer,\
                all_features, new_all_features, shap_explainer_name = self.getExplainableFile()

            def convert_to_lime_format(x, full_categorical_names, col_names=None, invert=False):
                if not isinstance(x, pd.DataFrame):
                    x_lime = pd.DataFrame(x, columns=col_names)
                else:
                    x_lime = x.copy()
                for k, v in full_categorical_names.items():
                    if not invert:
                        label_map = {
                            str_label: int_label for int_label, str_label in enumerate(v)}
                    else:
                        label_map = {
                            int_label: str_label for int_label, str_label in enumerate(v)}
                    x_lime.iloc[:, k] = x_lime.iloc[:, k].map(label_map)
                return x_lime

            def shap_and_eli5_custom_format(x):
                j_data = convert_to_lime_format(x, categorical_names)
                imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
                imp_mean.fit(j_data)
                custom_data = pd.DataFrame(imp_mean.fit_transform(j_data), columns=x.columns, index=j_data.index)
                if check_new_selected_features[0] != "no_selected_features":
                    return custom_data[new_selected_features]
                else:
                    data = x
                    ohe_data = pd.DataFrame(data_prepro.transform(data), columns=all_features)
                    return ohe_data

            with open(path, "r") as file:
                missing_values_formats = ['n/a', 'na', '--', '?', ' ', 'NA', 'N/A', 'NaN']
                reader = pd.read_csv(file, header=0, na_values=missing_values_formats, sep=None, engine='python',
                                     encoding='UTF-8')
                df = pd.DataFrame(data=reader)

            #
            def shap_imputed_data(x):
                if df.isna().any().any():
                    data = df
                    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
                    imp_mean.fit(data)
                    shap_imputed_data = pd.DataFrame(imp_mean.fit_transform(data), columns=x.columns, index=data.index)
                    return shap_imputed_data
                else:
                    return df

            if len(categorical_names) != 0:
                print("categorical_names: yes")
                shap_and_eli5_dataset = shap_and_eli5_custom_format(df)
            elif len(full_categorical_names) != 0:
                print("full_categorical_names: yes")
                shap_and_eli5_dataset = shap_and_eli5_custom_format(df)
            else:
                shap_and_eli5_dataset = shap_imputed_data(df)
                print("no: no")
            # run the shap_values file if exists
            shap_values_file = os.path.join(shap_values_folder_path, "shap_values_file.joblib")
            for filename in os.listdir(shap_values_folder_path):
                if fnmatch.fnmatch(filename, 'shap_values_file.joblib'):
                    with open(os.path.join(shap_values_folder_path, filename), 'rb') as f:
                        shap_values = joblib.load(f)
            # create the shap_values file if not exists
            shap_exist = exists(shap_values_file)
            if shap_exist:
                pass
            else:
                if fnmatch.fnmatch(shap_explainer_name, "shap_explainer"):
                    shap_values = shap_explainer.shap_values(shap_and_eli5_dataset)
                    joblib.dump(shap_values, filename=shap_values_file)
                if fnmatch.fnmatch(shap_explainer_name, "Tree_shap_explainer"):
                    shap_values = shap_explainer.shap_values(shap_and_eli5_dataset)
                    joblib.dump(shap_values, filename=shap_values_file)
                else:
                    pd.options.display.max_rows = None
                    pd.options.display.max_columns = None
                    print(shap_and_eli5_dataset)
                    shap_values = shap_explainer.shap_values(shap_and_eli5_dataset)
                    joblib.dump(shap_values, filename=shap_values_file)
            all_shap_observations = shap_and_eli5_dataset
            shap_observation = shap_and_eli5_dataset.iloc[[i], :]
            shap_observation_index = shap_and_eli5_dataset.index[i]
            # Get the class names/target names
            target_columns_file = os.path.join(pipes_path, "target_labels.pkl")
            target_columns_exist = os.path.isfile(target_columns_file)
            if target_columns_exist:
                with open(os.path.join(target_columns_file), "rb") as f:
                    target_columns = dill.load(f)
            if type(shap_values) == list:
                # plot 1
                if check_new_selected_features[0] != "no_selected_features":
                    pipeline, categorical_names, full_categorical_names, lime_explainer, data_prepro, shap_explainer,\
                    all_features, new_all_features, selected_features, shap_explainer_name, new_selected_features = \
                        self.getExplainableFile()
                    plot = shap.force_plot(shap_explainer.expected_value[0], shap_values[0],
                                           features=all_shap_observations, feature_names=selected_features, show=False,
                                           out_names=target_columns)

                    #shap.save_html(os.path.join(shap_values_folder_path, "shap_force_plot.htm"), plot)
                    shap_plot1_file = os.path.join(shap_values_folder_path, "shap_force_plot.htm")
                else:
                    plot = shap.force_plot(shap_explainer.expected_value[0], shap_values[0],
                                           features=all_shap_observations, feature_names=all_features, show=False)
                    shap.save_html(os.path.join(shap_values_folder_path, "shap_force_plot.htm"), plot)
                    shap_plot1_file = os.path.join(shap_values_folder_path, "shap_force_plot.htm")
                # plot 2 : single value
                plot2 = shap.force_plot(shap_explainer.expected_value[1], shap_values[1][i, :],
                                        features=shap_observation, feature_names=shap_and_eli5_dataset.columns,
                                        show=False)
                shap.save_html(os.path.join(shap_values_folder_path, "" + str(shap_observation_index)
                                            + "_row_shap_force.htm"), plot2)
                shap_plot2_single_value_file = os.path.join(shap_values_folder_path, "" + str(shap_observation_index)
                                                            + "_row_shap_force.htm")
            else:
                # plot 1
                if check_new_selected_features[0] != "no_selected_features":
                    pipeline, categorical_names, full_categorical_names, lime_explainer, data_prepro, shap_explainer, \
                    all_features, new_all_features, selected_features, shap_explainer_name, new_selected_features = \
                        self.getExplainableFile()
                    plot = shap.force_plot(shap_explainer.expected_value, shap_values, features=all_shap_observations,
                                           feature_names=selected_features, show=False)
                    shap.save_html(os.path.join(shap_values_folder_path, "shap_force_plot.htm"), plot)
                    shap_plot1_file = os.path.join(shap_values_folder_path, "shap_force_plot.htm")
                else:
                    plot = shap.force_plot(shap_explainer.expected_value, shap_values, features=all_shap_observations,
                                           feature_names=all_features, show=False)
                    shap.save_html(os.path.join(shap_values_folder_path, "shap_force_plot.htm"), plot)
                    shap_plot1_file = os.path.join(shap_values_folder_path, "shap_force_plot.htm")
                # plot 2 - single value
                plot2 = shap.force_plot(shap_explainer.expected_value, shap_values[i, :], features=shap_observation,
                                        feature_names=shap_and_eli5_dataset.columns, show=False)
                shap.save_html(os.path.join(shap_values_folder_path, "" + str(shap_observation_index)
                                            + "_row_shap_force.htm"), plot2)
                shap_plot2_single_value_file = os.path.join(shap_values_folder_path, "" + str(shap_observation_index)
                                                            + "_row_shap_force.htm")
            # SHAP Summary image
            fig_summary = plt.figure()
            if target_columns_exist:
                shap.summary_plot(shap_values, shap_and_eli5_dataset, feature_names=shap_and_eli5_dataset.columns,
                                  show=False, class_names=target_columns)
            else:
                shap.summary_plot(shap_values, shap_and_eli5_dataset, feature_names=shap_and_eli5_dataset.columns,
                                  show=False)
            fig_summary.savefig(os.path.join(shap_values_folder_path, 'shap_summary.png'), bbox_inches='tight', dpi=150)
            shap_summary_image = os.path.join(shap_values_folder_path, 'shap_summary.png')

            # fig_summary2 = plt.figure()
            # if target_columns_exist:
            #     shap.bar_plot(shap_values[0], shap_and_eli5_dataset,
            #                       show=False)
            # else:
            #     shap.bar_plot(shap_values[0], shap_and_eli5_dataset,
            #                       show=False)
            # fig_summary2.savefig(os.path.join(shap_values_folder_path, 'shap_summary2.png'), bbox_inches='tight', dpi=150)
            # shap_summary_image2 = os.path.join(shap_values_folder_path, 'shap_summary2.png')
            # Message Box open prediction
            if os.path.isfile(shap_plot2_single_value_file):
                msg_done = QMessageBox()
                msg_done.setWindowTitle("Prediction Finished!")
                msg_done.setIcon(QMessageBox.Question)
                msg_done.setText("Prediction done do you want to open the SHAP predictions?")
                # msg_done.setStandardButtons(QMessageBox.Open | QMessageBox.Cancel | QMessageBox.Yes | QMessageBox.No)
                msg_done.setStandardButtons(QMessageBox.Open | QMessageBox.Cancel | QMessageBox.Yes)
                open_btn = msg_done.button(QMessageBox.Open)
                open_btn.setText("Open Test Sample Plot")
                open1_btn = msg_done.button(QMessageBox.Yes)
                open1_btn.setText("Open SHAP Summary")
                # open2_btn = msg_done.button(QMessageBox.No)
                # open2_btn.setText("Open SHAP Force Summary")
                x = msg_done.exec_()
                if msg_done.clickedButton() == open_btn:
                    webbrowser.open(shap_plot2_single_value_file, new=2)
                if msg_done.clickedButton() == open1_btn:
                    im = Image.open(shap_summary_image)
                    im.show()
                # if msg_done.clickedButton() == open2_btn:
                #     webbrowser.open(shap_plot1_file, new=2)
                    # im2 = Image.open(shap_summary_image2)
                    # im2.show()
            return
        else:
            exception = QMessageBox()
            exception.setWindowTitle("Error")
            exception.setIcon(QMessageBox.Information)
            exception.setText('Error: Please add the Test dataset or the Sample Number.')
            x = exception.exec_()
            return None

    # #########################################
    # ############ BACK BUTTON ################
    # #########################################

    def back(self):
        from main_page import Ui_MainWindow
        main_page = QtWidgets.QMainWindow()
        main_page.ui = Ui_MainWindow()
        main_page.ui.setupUi(main_page)
        main_page.show()


# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     prediction = QtWidgets.QDialog()
#     ui = Ui_prediction()
#     ui.setupUi(prediction)
#     prediction.show()
#     sys.exit(app.exec_())
