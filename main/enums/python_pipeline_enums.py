from enum import Enum


class PythonPipelineEnums(Enum):
    pipeline_imports = """
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

from sklearn import set_config                      # to change the display
from sklearn.utils import estimator_html_repr       # to save the diagram into HTML format

from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score

# eli5 purposes
sel_fea = []
selected_features = []
new_selected_features = []

             """
    pipeline_comments = """
# creating the pipeline starting with the preprocessing algorithms of 
# numerical values then the preprocessing algorithms of categorical values,
# next we create the feature selection if there is any and in the end we add 
# the selected machine learning model.\n"""
    no_categorical_features_preprocessor = """preprocessor = ColumnTransformer(transformers=[
                                                         ('num', numeric_transformer, num_features),
                                                         ('cat', 'passthrought', cat_features)])\n\n"""
    numerical_features_preprocessor = """preprocessor = ColumnTransformer(transformers=[
                                                         ('num', 'passthrough', num_features),
                                                         ('cat', categorical_transformer, cat_features)])\n\n"""
    numerical_and_cateorical_features_preprocessor = """preprocessor = ColumnTransformer(transformers=[
                                                    ('num', numeric_transformer, num_features),
                                                    ('cat', categorical_transformer, cat_features)])\n\n"""
    classification_ml_problem = """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
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
                     """
    data_split = "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)"


class ReviewContextFromRatingEnum(Enum):
    GOOGLE_API_OR_WEB = "Review rating: out of 5, the score is {}, response only based on the review score."
    BOOKING_RAPIDAPI = "Review rating: out of 10, the score is {}, response only based on the review score."

    def format(self, review_rating):
        return self.value.format(review_rating)


class PipelineCodeWithMetricsEnum(Enum):
    gridsearch_select_but_use_random_search_because_more_than_9_params = """
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
pipeline.get_params("{model_name}")
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
    filename_scores = open(os.path.join(file_dir, "{model_name3}"), "w", encoding='utf-8')
    y_true = Y_test
    y_pred = pipeline.predict(X_test)
    # report = metrics.classification_report(y_true, y_pred, target_names=target_labels)
    filename_scores.write("Problem: Classification" + "\\n")
    filename_scores.write("model's name: " "{model_name}" + "\\n")
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
        "Target Column Name": [" "{target_column}" "],
        "Model's Name": ["{model_name}"],
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
    filename_scores_html = open(os.path.join(file_dir, "{model_name4}"), "w", encoding='utf-8')
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
filename = os.path.join(file_dir, "{model_name2}")
joblib.dump(pipeline, filename=filename)\n\n"""

    gridsearch_cv_less_or_equal_8_params = """
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
pipeline.get_params("{model_name}")
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
    filename_scores = open(os.path.join(file_dir, "{model_name3}"), "w", encoding='utf-8')
    y_true = Y_test
    y_pred = pipeline.predict(X_test)
    # report = metrics.classification_report(y_true, y_pred, target_names=target_labels)
    filename_scores.write("Problem: Classification" + "\\n")
    filename_scores.write("model's name: " "{model_name}" + "\\n")
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
        "Target Column Name": [" "{target_column}" "],
        "Model's Name": ["{model_name}"],
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
    filename_scores_html = open(os.path.join(file_dir, "{model_name4}"), "w", encoding='utf-8')
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
filename = os.path.join(file_dir, "{model_name2}")
joblib.dump(pipeline, filename=filename)\n\n"""
    random_search = """
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
pipeline.get_params("{model_name}")
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
    filename_scores = open(os.path.join(file_dir, "{model_name3}"), "w", encoding='utf-8')
    y_true = Y_test
    y_pred = pipeline.predict(X_test)
    # report = metrics.classification_report(y_true, y_pred, target_names=target_labels)
    filename_scores.write("Problem: Classification" + "\\n")
    filename_scores.write("model's name: " "{model_name}" + "\\n")
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
        "Target Column Name": [" "{target_column}" "],
        "Model's Name": ["{model_name}"],
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
    filename_scores_html = open(os.path.join(file_dir, "{model_name4}"), "w", encoding='utf-8')
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
filename = os.path.join(file_dir, "{model_name2}")
joblib.dump(pipeline, filename=filename)\n\n"""

    perm_importance = """
# #############################################
# ######### Permutation Importance ############
# #############################################


try:
    import eli5
    from eli5.sklearn import PermutationImportance
    eli5_data = pipeline.named_steps["preprocessor"].transform(X_test)
    permuter = PermutationImportance(pipeline.named_steps["{model_name}"],
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
    print("This model does not support ELI5 library.")"""