# import os
# import sys
# from PyQt5.QtWidgets import QMessageBox, QFileDialog
# from PyQt5 import QtWidgets
#
#
# class MessageBox(QMessageBox):
#
#     def __init__(self, parent=None):
#         super(MessageBox, self).__init__(parent)
#         self.main_path_list = []
#
#     def message_box(self):
#         main_path_ = []
#         msg2 = QMessageBox()
#         msg2.setWindowTitle("Save Project")
#         msg2.setIcon(QMessageBox.Question)
#         msg2.setText("Do you want to save the new project on the Desktop or Somewhere else?")
#         desktop_button = msg2.addButton("Desktop", QtWidgets.QMessageBox.YesRole)
#         somewhere_else_btn = msg2.addButton("Somewhere Else", QtWidgets.QMessageBox.NoRole)
#         cancel_btn = msg2.addButton("Cancel", QtWidgets.QMessageBox.NoRole)
#         x2 = msg2.exec_()
#         if msg2.clickedButton() == somewhere_else_btn:
#             dialog = QFileDialog()
#             filepath = dialog.getSaveFileName(os.getenv('Home'))
#             if not filepath[0]:
#                 main_path_.append(None)
#                 self.main_path_list.append(None)
#                 return None
#             else:
#                 main_path = filepath[0]
#                 main_path_.append(filepath[0])
#                 self.main_path_list.append(filepath[0])
#                 print(main_path)
#         elif msg2.clickedButton() == desktop_button:
#             self.main_path_list.append("Desktop")
#         elif msg2.clickedButton() == cancel_btn:
#             self.main_path_list.append("Cancel")
#         return main_path_
#
#     sys.stdout.flush()
import os
from PyQt5.QtWidgets import QMessageBox, QFileDialog


class MessageBox:

    def __init__(self):
        self.main_path_list = []

    def show_message_box(self):
        response = self.ask_save_location()
        if response == "Somewhere Else":
            return self.select_custom_path()
        elif response == "Desktop":
            self.main_path_list.append("Desktop")
        elif response == "Cancel":
            self.main_path_list.append("Cancel")
        else:
            self.main_path_list.append(None)

    @staticmethod
    def ask_save_location():
        msg = QMessageBox()
        msg.setWindowTitle("Save Project")
        msg.setIcon(QMessageBox.Question)
        msg.setText("Do you want to save the new project on the Desktop or Somewhere else?")
        msg.addButton("Desktop", QMessageBox.YesRole)
        msg.addButton("Somewhere Else", QMessageBox.NoRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.NoRole)

        msg.exec_()
        clicked_button = msg.clickedButton()

        if clicked_button == cancel_btn:
            return "Cancel"
        else:
            return clicked_button.text()

    def select_custom_path(self):
        dialog = QFileDialog()
        filepath, _ = dialog.getSaveFileName(caption="Select Save Location", directory=os.getenv('Home'))

        if filepath:
            self.main_path_list.append(filepath)
            return filepath
        else:
            self.main_path_list.append(None)
            return None
