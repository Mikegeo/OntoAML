import os
from datetime import datetime

import pandas as pd
from PyQt5.QtWidgets import QMessageBox
from ydata_profiling import ProfileReport

from main.create.message_box import MessageBox


class DataProfiler:

    def __init__(self, training_ui):
        self.training_ui = training_ui

    def datasetDetailsProfiling(self):
        if not self.confirmation_dialog("Dataset",
                                        "Checking dataset's information it might take some time please press OK to continue"):
            return

        main_path_list = self.get_main_path_list()
        if not main_path_list or main_path_list[0] == "Cancel":
            return

        if not self.confirmation_dialog("Dataset Profiling", "The system will inform you at the end of the process"):
            return

        path = self.training_ui.datasetPathLineEdit.text()
        dt_string = self.date_time()
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

        try:
            df = self.load_dataset(path)
            profile = self.generate_profile(df)
            main_folder = self.determine_main_folder(main_path_list, desktop, dt_string)
            self.save_profile(profile, main_folder)
            self.show_completion_message()
            self.training_ui.openDatasetProfileBtn.setEnabled(True)
        except Exception as e:
            self.show_error_message(str(e))

    @staticmethod
    def confirmation_dialog(title, text):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        return msg.exec_() == QMessageBox.Ok

    @staticmethod
    def get_main_path_list():
        p = MessageBox()
        p.show_message_box()
        return p.main_path_list

    @staticmethod
    def load_dataset(path):
        missing_values_formats = ["n/a", "na", "--", "?", " ", "NA", "N/A"]
        with open(path, "r") as file:
            return pd.read_csv(file, header=0, sep=None, na_values=missing_values_formats, engine='python',
                               encoding='UTF-8')

    @staticmethod
    def generate_profile(df):
        if len(df) > 10000 or len(df.columns) > 150:
            return ProfileReport(df, minimal=True, html={'style': {'full_width': True}})
        else:
            return ProfileReport(df, html={'style': {'full_width': True}})

    def determine_main_folder(self, main_path_list, desktop, dt_string):
        dataset_name = os.path.splitext(os.path.basename(self.training_ui.datasetPathLineEdit.text()))[0]
        if main_path_list[0] == "Desktop":
            folder = os.path.join(desktop, f"Main_Pipeline_{dataset_name}_{dt_string}")
        else:
            folder = main_path_list[0]
        self.training_ui.newProjectsPathLineEdit.setText(folder)
        return folder

    def save_profile(self, profile, main_folder):
        dataset_name = os.path.splitext(os.path.basename(self.training_ui.datasetPathLineEdit.text()))[0]
        profile_folder = os.path.join(main_folder, f"{dataset_name}_profile")

        if not os.path.exists(profile_folder):
            os.makedirs(profile_folder)

        html_file = os.path.join(profile_folder, "your_report.html")
        json_file = os.path.join(profile_folder, "your_report.json")

        profile.to_file(html_file)
        profile.to_file(json_file)

    def show_completion_message(self):
        msg = QMessageBox()
        msg.setWindowTitle("Dataset")
        msg.setIcon(QMessageBox.Information)
        msg.setText("The dataset's details are ready")
        msg.buttonClicked.connect(self.training_ui.datasetDetails)
        msg.exec_()

    @staticmethod
    def show_error_message(error_message):
        QMessageBox.critical(None, "Error", error_message)

    @staticmethod
    def date_time():
        return datetime.now().strftime("_date_%d_%m_%Y_time_%H_%M")
