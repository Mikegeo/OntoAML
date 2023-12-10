import os
import json
from PyQt5.QtCore import Qt


class DataAnalyzer:

    def __init__(self, training_ui):
        self.training_ui = training_ui

    def dataset_details(self):
        dataset_name = self._get_dataset_name()
        data = self._load_data(dataset_name)

        name_column, type_column = self._extract_columns(data)
        missing_values_number, sample_size, feature_size = self._extract_table_details(data)
        uniform_unique_columns = self._extract_uniform_unique_columns(data)
        numerical_columns, categorical_columns = self._categorize_columns(name_column, type_column)
        cat_columns_with_num_values_names, cat_columns_with_num_values, columns_with_high_missing_values_number, \
            num_columns_with_cat_values = self._additional_column_checks(data)

        self._update_ui(feature_size, sample_size, numerical_columns, categorical_columns)
        column_list = self._populate_comboboxes(numerical_columns, categorical_columns)

        categorical_columns_with_numerical_values = self._handle_numerical_categorical_overlap(
            num_columns_with_cat_values, numerical_columns, categorical_columns)

        return sample_size, feature_size, numerical_columns, categorical_columns, missing_values_number, dataset_name,\
            categorical_columns_with_numerical_values, column_list, num_columns_with_cat_values

    def _get_dataset_name(self):
        path = self.training_ui.datasetPathLineEdit.text()
        dataset_name_ext = os.path.basename(path)
        return os.path.splitext(dataset_name_ext)[0]

    def _load_data(self, dataset_name):
        main_folder = self.training_ui.newProjectsPathLineEdit.text()
        with open(f"{main_folder}/{dataset_name}_profile/your_report.json") as data_file:
            return json.load(data_file)

    def _populate_comboboxes(self, numerical_columns, categorical_columns):
        column_list = numerical_columns + categorical_columns
        self.training_ui.comboBoxDeleteColumns.addItem("None")
        self.training_ui.comboBoxDeleteColumns.addItems(column_list)
        for i in range(len(column_list)):
            item = self.training_ui.comboBoxDeleteColumns.model().item(i, 0)
            item.setCheckState(Qt.Unchecked)
        return column_list

    @staticmethod
    def _extract_columns(data):
        name_column = []
        type_column = []
        for key, value in data.get('variables', {}).items():
            name_column.append(key)
            type_column.append(value.get('type', None))
        return name_column, type_column

    @staticmethod
    def _extract_table_details(data):
        table_data = data.get('table', {})
        missing_values_number = table_data.get('n_cells_missing', "")
        sample_size = table_data.get('n', "")
        feature_size = table_data.get('n_var', "")
        return missing_values_number, sample_size, feature_size

    @staticmethod
    def _extract_uniform_unique_columns(data):
        uniform_unique_columns = []
        for item, item1 in data.get('variables', {}).items():
            if item1.get('is_unique', False):
                uniform_unique_columns.append(item)
        return uniform_unique_columns

    @staticmethod
    def _categorize_columns(name_column, type_column):
        numerical_columns = []
        categorical_columns = []
        for name, col_type in zip(name_column, type_column):
            if col_type in ["Categorical", "Boolean"]:
                categorical_columns.append(name)
            else:
                numerical_columns.append(name)
        return numerical_columns, categorical_columns

    @staticmethod
    def _additional_column_checks(data):
        cat_columns_with_num_values_names = []
        cat_columns_with_num_values = []
        num_columns_with_cat_values = []
        columns_with_high_missing_values_number = []

        for item, item1 in data.get('variables', {}).items():
            if isinstance(item1, dict):
                for element1, element2 in item1.items():
                    # Check for categorical columns with numerical values
                    if element1 == "category_alias_values":
                        cat_columns_with_num_values_names.append(item)
                        if isinstance(element2, dict):
                            if all(value == "Decimal_Number" for value in element2.values()) or \
                                    all(value in ["Decimal_Number", "Other_Punctuation"] for value in
                                        element2.values()):
                                cat_columns_with_num_values.append("yes")
                            else:
                                cat_columns_with_num_values.append("no")

                    # Check for columns with high missing values
                    if element1 == "p_missing" and element2 > 0.95:
                        columns_with_high_missing_values_number.append(item)

                    # Check if there is a column which has numerical values but should be considered as categorical
                    if element1 == "type" and element2 == "Numeric":
                        for element3, element4 in item1.items():
                            if element3 in ["p_distinct", "n_distinct"] and element4 < 0.000099:
                                for value_counts in item1.get("value_counts_without_nan", {}).keys():
                                    if value_counts.replace('.', '', 1).isdigit() and len(value_counts) == 1:
                                        num_columns_with_cat_values.append(item)
                                        break  # Break if condition is met to avoid duplicates

        return cat_columns_with_num_values_names, cat_columns_with_num_values, columns_with_high_missing_values_number, num_columns_with_cat_values

    def _update_ui(self, feature_size, sample_size, numerical_columns, categorical_columns):
        self.training_ui.textViewFeatureSize.setText(str(feature_size))
        self.training_ui.textViewSampleSize.setText(str(sample_size))
        for target_column in numerical_columns:
            self.training_ui.comboBoxTargetColumn.addItem(target_column)
        for target_column in categorical_columns:
            self.training_ui.comboBoxTargetColumn.addItem(target_column)

    @staticmethod
    def _handle_numerical_categorical_overlap(num_columns_with_cat_values, numerical_columns,
                                              categorical_columns):
        numerical_columns_with_categorical_values = []
        for item in num_columns_with_cat_values:
            if item not in numerical_columns_with_categorical_values:
                numerical_columns_with_categorical_values.append(item)
            if item in numerical_columns:
                numerical_columns.remove(item)
                categorical_columns.append(item)
        # Additional logic can be added here as needed
        return numerical_columns_with_categorical_values



# Example of how you might initialize and use this class:
# analyzer = DataAnalyzer(datasetPathLineEdit, newProjectsPathLineEdit, textViewFeatureSize, textViewSampleSize, comboBoxTargetColumn, comboBoxDeleteColumns)
# results = analyzer.dataset_details()
