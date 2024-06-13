# Standard Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import shap
import lime
import dill

from sklearn import set_config  # to change the display
from sklearn.utils import estimator_html_repr  # to save the diagram into HTML format

from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score

# eli5 purposes
sel_fea = []
selected_features = []
new_selected_features = []

# Imports from the selected algorithms
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

missing_values_formats = ['n/a', 'na', '--', '?', ' ', 'NA', 'N/A']
separation = "[,]+|;|:"
df = pd.read_csv('path_to_csv.csv',
                 na_values=missing_values_formats, sep=None, engine='python', encoding='UTF-8')
df = df.dropna(subset=['target'])
X = df.drop(['target', ], axis=1)
y = df['target']
num_features = ['age', 'SGOT',  ]
cat_features = ['drug', 'sex', ]
categorical_features_with_numerical_values = []
df[num_features] = df[num_features].apply(pd.to_numeric, errors='coerce')
df[cat_features] = df[cat_features].astype(object)

# call the names of the selected machine learning algorithms
SimpleImputer = SimpleImputer
OneHotEncoder = OneHotEncoder
StandardScaler = StandardScaler()
RandomForestClassifier = RandomForestClassifier
# Adjust the categorical transformer to not include 'presence_of_edema'
cat_features_without_column = [feature for feature in cat_features if feature != 'column']
# creating the pipeline starting with the preprocessing algorithms of
# numerical values then the preprocessing algorithms of categorical values,
# next we create the feature selection if there is any and in the end we add
# the selected machine learning model.
numeric_transformer = Pipeline(steps=[
    ('SimpleImputer', SimpleImputer(strategy='median')),
    ('StandardScaler', StandardScaler),
])

categorical_transformer = Pipeline(steps=[
    ('SimpleImputer', SimpleImputer(strategy='most_frequent')),
    ('OneHotEncoder', OneHotEncoder(sparse=False, handle_unknown='ignore', drop='if_binary')),
])

# New transformer for the ordinal feature if needed
ordinal_features = ['ordinal_feature']
ordinal_transformer = Pipeline(steps=[
    ('SimpleImputer', SimpleImputer(strategy='most_frequent')),
    ('OrdinalEncoder', OrdinalEncoder(categories=[['0', '0.5', '1']])),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features_without_column),
        ('ord', ordinal_transformer, ordinal_features),
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('RandomForestClassifier', RandomForestClassifier()),
])

param_grid = {
    'RandomForestClassifier__criterion': ['entropy'],
    'RandomForestClassifier__max_features': [0.1, 0.15000000000000002],
    'RandomForestClassifier__min_samples_split': [8],
    'RandomForestClassifier__min_samples_leaf': [5, 6],
    'RandomForestClassifier__bootstrap': [True, False],
}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
labelencoder_Y = LabelEncoder()
labelencoder_Y.fit(y)
Y_train = labelencoder_Y.transform(y_train)
Y_test = labelencoder_Y.transform(y_test)
Y_all = labelencoder_Y.transform(y)

# create target_labels for interpretability
target_names = labelencoder_Y.inverse_transform(Y_all)
target_labels = list(dict.fromkeys(target_names))
file_dir = os.path.dirname(__file__)
target_labels_file = open(os.path.join(file_dir, 'target_labels.pkl'), "wb")
dill.dump(target_labels, target_labels_file)

# Grid search implentation
import time
from itertools import product
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

# Set the time budget (in seconds)
time_budget = 3600

# Set the start time
start_time = time.time()

# Initialize the best score and best parameters
best_score = -float("inf")
best_params = {}
best_estimator = None
std_score = -float("inf")
min_score = -float("inf")
max_score = -float("inf")
scores_var = -float("inf")

# Flatten the parameter grid
flattened_param_grid = {}
for key, value_list in param_grid.items():
    if not isinstance(value_list, (list, tuple)):
        value_list = [value_list]
    flattened_param_grid[key] = value_list

# Get the list of parameter combinations
param_combinations = list(product(*flattened_param_grid.values()))

# Set the scoring metric
scoring = "accuracy"

# Set the verbose level
verbose = 4

# Iterate over the parameter combinations
for params in param_combinations:
    # Set the pipeline's hyperparameters
    params_dict = dict(zip(flattened_param_grid.keys(), params))
    pipeline.set_params(**params_dict)

    # Get the cross-validation score for this parameter combination
    try:
        cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=kf, n_jobs=3, scoring=scoring, verbose=verbose)
        mean_score = cv_scores.mean()
        std_score = np.std(cv_scores)
        min_score = np.min(cv_scores)
        max_score = np.max(cv_scores)
        scores_var = np.var(cv_scores)
    except:
        # An error occurred, print a message and continue with the next iteration
        print("An error occurred, continuing with next iteration")
        continue

    print(f"Parameters: {params_dict}")

    # If the score is better than the current best score, update the best score and best parameters
    if mean_score > best_score:
        best_score = mean_score
        best_params = params
        best_estimator = pipeline
        std_score = std_score
        min_score = min_score
        max_score = max_score
        scores_var = scores_var

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # If the time budget has been exceeded, break out of the loop
    if elapsed_time > time_budget:
        break

# Record the end time
end_time = time.time()
# Calculate the total elapsed time in seconds
total_elapsed_time_seconds = end_time - start_time
# Convert elapsed time to minutes
total_elapsed_time_minutes = total_elapsed_time_seconds / 60
# Print the elapsed time in minutes
print(f"Total elapsed time: {total_elapsed_time_minutes:.2f} minutes")

# Set the best parameters on the pipeline
best_params_dict = dict(zip(flattened_param_grid.keys(), best_params))
best_estimator.set_params(**best_params_dict)

# Fit the best estimator to the data
best_estimator.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score

# Predict the labels for the test data
y_pred = best_estimator.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(Y_test, y_pred)
print(f'Accuracy scikit-learn: {accuracy:.5f}')
print(f'Accuracy variance: {scores_var:.5f}')
print(f'std score: {std_score:.5f}')
print(f'best score: {best_score:.5f}')
print("Range of scores: {:.3f} - {:.3f}".format(min_score, max_score))
# Compare the standard deviation to the mean score and range of scores
if std_score < 0.1 * best_score:
    print("The scores are relatively consistent.")
else:
    print("The scores are relatively inconsistent.")
# Compare the mean and variance of the cross-validation scores
if scores_var > 0.05 and best_score > accuracy:
    print(
        "The model is overfitting, as the variance of the cross-validation scores is high and the mean is higher than the validation accuracy.")
else:
    print("The model is not overfitting.")
###########################################################################
###########################################################################


file_dir = os.path.dirname(__file__)
pipeline.set_params(**best_params_dict)
pipeline.get_params('RandomForestClassifier')
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
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20,
            correct_orientation=False, cmap='RdBu'):
    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

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
    plt.xlim((0, AUC.shape[1]))

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
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2: (len(lines) - 4)]:
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
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height,
            correct_orientation, cmap=cmap)


y_true = Y_test
y_pred = pipeline.predict(X_test)
if (target_labels == [1, 0]) or (target_labels == [0, 1]) or (target_labels == [1, 2]) or (target_labels == [2, 1]) or (
        target_labels == [-1, 1]) or (target_labels == [1, -1]):
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
    filename_scores = open(os.path.join(file_dir, 'RandomForestClassifier_trained_pipeline_scores.txt'), "w",
                           encoding='utf-8')
    y_true = Y_test
    y_pred = pipeline.predict(X_test)
    # report = metrics.classification_report(y_true, y_pred, target_names=target_labels)
    filename_scores.write("Problem: Classification" + "\n")
    filename_scores.write("model's name: " 'RandomForestClassifier' + "\n")
    filename_scores.write(report + "\n")
    filename_scores.write("Accuracy score: {:0.5f}".format(metrics.accuracy_score(y_true, y_pred)))
    filename_scores.write("\n")
    filename_scores.write("F1 score: {:0.3f}".format(metrics.f1_score(y_true, y_pred)))
    filename_scores.write("\n")
    filename_scores.write("Auc score: {:0.3f}".format(metrics.roc_auc_score(Y_test, y_pred)) + "\n")
    filename_scores.write("Overfit estimation with different {:?f}" + "\n")
    filename_scores.write("{:0.1f}".format(overfit_estimation) + "\n")
    filename_scores.write("{:0.2f}".format(overfit_estimation) + "\n")
    filename_scores.write("{:0.3f}".format(overfit_estimation) + "\n")
    filename_scores.write("{:0.4f}".format(overfit_estimation) + "\n")
    filename_scores.write("Learning curve with different {:?f} for the overall mean value" + "\n")
    filename_scores.write("{:0.1f}".format(np.mean(overfit_gap_learning_curve)) + "\n")
    filename_scores.write("{:0.2f}".format(np.mean(overfit_gap_learning_curve)) + "\n")
    filename_scores.write("{:0.3f}".format(np.mean(overfit_gap_learning_curve)) + "\n")
    filename_scores.write("{:0.4f}".format(np.mean(overfit_gap_learning_curve)) + "\n")
    filename_scores.write("Learning curve with different {:?f} for the last value" + "\n")
    filename_scores.write("{:0.1f}".format(last_overfit_gap) + "\n")
    filename_scores.write("{:0.2f}".format(last_overfit_gap) + "\n")
    filename_scores.write("{:0.3f}".format(last_overfit_gap) + "\n")
    filename_scores.write("{:0.4f}".format(last_overfit_gap) + "\n")
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
        "Target Column Name": [" target "],
        "Model's Name": ['RandomForestClassifier'],
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
    filename_scores_html = open(os.path.join(file_dir, 'RandomForestClassifier_trained_pipeline_scores.html'), "w",
                                encoding='utf-8')
    filename_scores_html.write(custom_report_html)
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Classification Report:" + "<br>")
    filename_scores_html.write(report_html + "<br>")
    filename_scores_html.write('<img src = "' + classification_report_plot + '" alt ="cfg"><br>')
    filename_scores_html.write("<font size= "'6'"><b> " + "Roc Auc curve figure:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + roc_curve_plot + '" alt ="cfg"><br>')
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Overfit Report:" + "<br>")
    filename_scores_html.write(custom_overfit_report + "<br>")
    filename_scores_html.write(
        "<font size= "'6'"><b> " + "Learning Curve - Overfitting or Underfitting:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + overfitting_plot + '" alt ="cfg">')
    filename_scores_html.close()


print_report(pipeline)
filename = os.path.join(file_dir, 'RandomForestClassifier_trained_pipeline.joblib')
joblib.dump(pipeline, filename=filename)

data_preprocessor = os.path.join(file_dir, 'Data_preprocessor.joblib')
joblib.dump(pipeline.named_steps["preprocessor"], filename=data_preprocessor)

preprocessor = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
ohe_categories = preprocessor.named_steps["OneHotEncoder"].categories_
new_ohe_features = [f"{col}__{val}" for col, vals in zip(cat_features, ohe_categories) for val in vals]
all_features = num_features + new_ohe_features
all_features_file = open(os.path.join(file_dir, 'all_features'), 'wb')
dill.dump(all_features, all_features_file)

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
    # Converts data with categorical values as string into the right format
    # for LIME, with categorical values as integers labels.
    #
    # It takes categorical_names, the same dictionary that has to be passed
    # to LIME to ensure consistency.
    #
    # col_names and invert allow to rebuild the original dataFrame from
    # a numpy array in LIME format to be passed to a Pipeline or sklearn
    # OneHotEncoder
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
    # Convert to the format expected by LIME, which may not necessarily align with the original column count
    lime_data = convert_to_lime_format(X, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    # Perform the imputation
    imputed_data = imp_mean.fit_transform(lime_data)
    # Since the imputation should not change the number of columns, we adjust how columns are assigned
    # Ensure that the columns attribute is correctly set based on the output shape
    columns_after_imputation = X.columns[:imputed_data.shape[1]]
    imputed_lime_data = pd.DataFrame(imputed_data, columns=columns_after_imputation, index=lime_data.index)
    return imputed_lime_data


imputed_lime_X_train = convert_to_imputed_lime(X_train)

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

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer


def get_feature_names(column_transformer):
    """
    Get feature names from all transformers within a ColumnTransformer.
    """
    output_features = []

    # Loop through each transformer within the ColumnTransformer
    for name, pipe, features in column_transformer.transformers_:
        if name == 'remainder' and pipe == 'drop':
            continue  # If 'remainder' is set to 'drop', skip it
        if hasattr(pipe, 'get_feature_names_out'):
            # Get the feature names from the transformer
            transformer_features = pipe.get_feature_names_out(features)
        else:
            # If the transformer does not support 'get_feature_names_out',
            # use the provided feature names directly
            transformer_features = features
        output_features.extend(transformer_features)

    return output_features


# Correctly access the ColumnTransformer from within the pipeline
column_transformer = pipeline.named_steps['preprocessor']
all_features = get_feature_names(column_transformer)
all_features_file = open(os.path.join(file_dir, 'all_features'), 'wb')
dill.dump(all_features, all_features_file)


def shap_and_eli5_custom_format(x):
    from sklearn.impute import SimpleImputer
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    # Perform the imputation
    custom_data_transformed = imp_mean.fit_transform(j_data)
    # Adjust the columns parameter to match the actual data shape after imputation
    columns_after_imputation = x.columns[:custom_data_transformed.shape[1]]
    custom_data = pd.DataFrame(custom_data_transformed, columns=columns_after_imputation, index=j_data.index)

    if len(new_selected_features) > 0:
        return custom_data[new_selected_features]
    else:
        # Ensure that the OHE data transformation step is correctly handling the shape
        # Transform x using the preprocessor pipeline, ensuring all preprocessing steps are applied
        preprocessed_data = pipeline.named_steps['preprocessor'].transform(x)

        # Generate feature names from the ColumnTransformer within the pipeline
        feature_names = get_feature_names(pipeline.named_steps['preprocessor'])

        # Create a DataFrame with the transformed data and correct feature names
        ohe_data = pd.DataFrame(preprocessed_data, columns=feature_names, index=x.index)

    return ohe_data


# #############################################
# ######### SHAP VALUES ############
# #############################################
shap_data = shap_and_eli5_custom_format(X_train)

shap.initjs()
shap_explainer = shap.TreeExplainer(pipeline.named_steps['RandomForestClassifier'])
shap_explainer_file = open(os.path.join(file_dir, 'Tree_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)
shap_and_eli5_dataset = shap_and_eli5_custom_format(X_test)
shap_values = shap_explainer.shap_values(shap_and_eli5_dataset)
fig_summary = plt.figure()
shap.summary_plot(shap_values[1], shap_and_eli5_dataset, feature_names=shap_and_eli5_dataset.columns,
                  show=False, class_names=target_labels)
fig_summary.savefig(os.path.join(file_dir, 'shap_summary_target_1.png'), bbox_inches='tight', dpi=150)

fig_summary_5 = plt.figure()
shap.summary_plot(shap_values, shap_and_eli5_dataset, feature_names=shap_and_eli5_dataset.columns,
                  show=False, class_names=target_labels)
fig_summary_5.savefig(os.path.join(file_dir, 'shap_summary_all.png'), bbox_inches='tight', dpi=150)


def shap_imputed_data_2(df):
    # Check if there are any missing values in the dataframe
    if df.isna().any().any():
        # Create an empty dataframe with the same structure as the input
        imputed_data = pd.DataFrame(index=df.index, columns=df.columns)

        # Process each column separately
        for column in df.columns:
            # Determine the data type of the column
            if df[column].dtype in ['int64', 'float64']:
                # Numerical data: Use median for imputation
                imp = SimpleImputer(missing_values=np.nan, strategy='median')
            else:
                # Categorical data: Use most frequent value for imputation
                imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

            # Fit the imputer and transform the data
            imputed_data[column] = imp.fit_transform(df[[column]])

        # Return the imputed dataframe
        return imputed_data
    else:
        # If there are no missing values, return the original dataframe
        return df


shap_df = pd.DataFrame(X_test, columns=X.columns)
shap_and_eli5_dataset_2 = shap_imputed_data_2(shap_df)
print(shap_and_eli5_dataset_2.head())
target_class_name = target_labels[1]
for feature in shap_and_eli5_dataset_2.columns:
    # Create figure with constrained layout
    plt.figure(figsize=(200, 222), constrained_layout=True)
    plt.title(f"SHAP Dependence Plot for {feature} (Target: {target_class_name})", pad=20)

    shap.dependence_plot(feature, shap_values[1], shap_and_eli5_dataset_2, interaction_index=None,
                         display_features=shap_and_eli5_dataset_2,
                         feature_names=shap_and_eli5_dataset_2.columns, show=False)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    filename = f"shap_dependence_plot_{feature}.png"
    file_path = os.path.join(file_dir, filename)
    plt.savefig(file_path, bbox_inches='tight', dpi=150)

    # Close the figure to free memory
    plt.close()

# #############################################
# ######### Permutation Importance ############
# #############################################

# try:
import eli5
from eli5.sklearn import PermutationImportance

eli5_data = shap_and_eli5_custom_format(X_test)
permuter = PermutationImportance(pipeline.named_steps['RandomForestClassifier'],
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
