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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

missing_values_formats = ['n/a', 'na', '--', '?', ' ', 'NA', 'N/A']
separation = "[,]+|;|:"
df = pd.read_csv(r"path_to_csv.csv",
                 na_values=missing_values_formats, sep=None, engine='python', encoding='UTF-8')
df = df.dropna(subset=['target'])
X = df.drop(['target', ], axis=1)
y = df['target']
num_features = ['Age', 'Height', 'Weight']
cat_features = ['cat1', 'cat2']
categorical_features_with_numerical_values = ['features with values like 0-1 or 1-2']
df[num_features] = df[num_features].apply(pd.to_numeric, errors='coerce')
df[cat_features] = df[cat_features].astype(object)

# call the names of the selected machine learning algorithms
SimpleImputer = SimpleImputer
OneHotEncoder = OneHotEncoder
StandardScaler = StandardScaler()
RandomForestRegressor = RandomForestRegressor

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
    ('StandardScaler', StandardScaler),
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('RandomForestRegressor', RandomForestRegressor()),
])

# This is the hyper-parameter search space based on the above algorithms
param_grid = {
    'RandomForestRegressor__max_features': [0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4,
                                            0.45, 0.5, 0.55, 0.6000000000000001, 0.6500000000000001, 0.7000000000000001,
                                            0.7500000000000001, 0.8, 0.8500000000000001, 0.9000000000000001,
                                            0.9500000000000001, 1.0],
    'RandomForestRegressor__min_samples_split': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20],
    'RandomForestRegressor__min_samples_leaf': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20],
    'RandomForestRegressor__bootstrap': [True, False],
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
file_dir = os.path.dirname(__file__)

# Grid search implementation
import time
from itertools import product
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

n = 5
kf = KFold(n_splits=n)

# Set the time budget (in seconds)
time_budget = 5100

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
scoring = "r2"

# Set the verbose level
verbose = n

# Iterate over the parameter combinations
for params in param_combinations:
    # Set the pipeline's hyperparameters
    params_dict = dict(zip(flattened_param_grid.keys(), params))
    pipeline.set_params(**params_dict)

    # Get the cross-validation score for this parameter combination
    try:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, n_jobs=2, scoring=scoring, verbose=verbose)
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
best_estimator.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = best_estimator.predict(X_test)

print(f'std score: {std_score:.5f}')
print(f'best score: {best_score:.5f}')
print("Range of scores: {:.3f} - {:.3f}".format(min_score, max_score))
# Compare the standard deviation to the mean score and range of scores
if std_score < 0.1 * best_score:
    print("The scores are relatively consistent.")
else:
    print("The scores are relatively inconsistent.")

pipeline.set_params(**best_params_dict)
pipeline.get_params('RandomForestRegressor')
pipeline.fit(X_train, y_train)
###########################################################################

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
plt.xlabel('Number of training example_pipeline_templates')
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
    # median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    # TXT Report
    filename_scores = open(os.path.join(file_dir, 'RandomForestRegressor_trained_pipeline_scores.txt'), "w",
                           encoding='utf-8')
    filename_scores.write("Problem: Regression" + "\n")
    filename_scores.write("model's name: " 'RandomForestRegressor' + "\n")
    filename_scores.write('explained_variance: ' + str(round(explained_variance, 4)) + "\n")
    # filename_scores.write('mean_squared_log_error: ' + str(round(mean_squared_log_error, 4)) + "\n")
    filename_scores.write('r2: ' + str(round(r2, 4)) + "\n")
    filename_scores.write('MAE: ' + str(round(mean_absolute_error, 4)) + "\n")
    filename_scores.write('MSE: ' + str(round(mse, 4)) + "\n")
    filename_scores.write('RMSE: ' + str(round(np.sqrt(mse), 4)) + "\n")
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
                 "Target Column Name": [" target "],
                 "model's name": ['RandomForestRegressor'],
                 "explained_variance": [str(round(explained_variance, 4))],
                 # "Mean Squared Log Error": [str(round(mean_squared_log_error, 4))],
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
    filename_scores_html = open(os.path.join(file_dir, 'RandomForestRegressor_trained_pipeline_scores.html'), "w",
                                encoding='utf-8')
    filename_scores_html.write(report_html + "<br>")
    filename_scores_html.write("<font size= "'6'"><p><b> " + "Overfit Report:" + "<br>")
    filename_scores_html.write(custom_overfit_report + "<br>")
    filename_scores_html.write(
        "<font size= "'6'"><b> " + "Learning Curve - Overfitting or Underfitting:" + "</b></font><br>")
    filename_scores_html.write('<img src = "' + overfitting_plot + '" alt ="cfg">')
    filename_scores_html.close()


y_true = y_test
y_pred = pipeline.predict(X_test)
regression_results(y_true, y_pred)
filename = os.path.join(file_dir, 'RandomForestRegressor_trained_pipeline.joblib')
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
                                                        mode="regression",
                                                        feature_names=X_train.columns.tolist(),
                                                        class_names=" target ",
                                                        categorical_names=categorical_names,
                                                        categorical_features=categorical_names.keys(),
                                                        discretize_continuous=False,
                                                        kernel_width=5,
                                                        random_state=0)

imputed_lime_X_test = convert_to_imputed_lime(X_test)

# save explanation to a dill file
with open('lime_explainer_file', 'wb') as f:
    dill.dump(lime_explainer, f)

from sklearn.impute import SimpleImputer


def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    custom_data_transformed = imp_mean.fit_transform(j_data)
    columns_after_imputation = x.columns[:custom_data_transformed.shape[1]]
    custom_data = pd.DataFrame(custom_data_transformed, columns=columns_after_imputation, index=j_data.index)

    if len(new_selected_features) > 0:
        return custom_data[new_selected_features]
    else:
        preprocessed_data = pipeline.named_steps['preprocessor'].transform(x)
        feature_names = get_feature_names(pipeline.named_steps['preprocessor'])
        ohe_data = pd.DataFrame(preprocessed_data, columns=feature_names, index=x.index)
    return ohe_data


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


# Fix deprecated np.bool before calling shap_explainer
# #############################################
# ######### SHAP VALUES ############
# #############################################
shap_data = shap_and_eli5_custom_format(X_train)

shap.initjs()
shap_explainer = shap.TreeExplainer(pipeline.named_steps['RandomForestRegressor'])
shap_explainer_file = open(os.path.join(file_dir, 'Tree_shap_explainer'), "wb")
dill.dump(shap_explainer, shap_explainer_file)

shap_and_eli5_dataset = shap_and_eli5_custom_format(X_test)
shap_values = shap_explainer.shap_values(shap_and_eli5_dataset)

fig_summary_5 = plt.figure()
shap.summary_plot(shap_values, shap_and_eli5_dataset, feature_names=shap_and_eli5_dataset.columns, show=False)
fig_summary_5.savefig(os.path.join(file_dir, 'shap_summary_all.png'), bbox_inches='tight', dpi=150)


from sklearn.impute import SimpleImputer


def shap_and_eli5_custom_format(x):
    j_data = convert_to_lime_format(x, categorical_names)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    custom_data_transformed = imp_mean.fit_transform(j_data)
    columns_after_imputation = x.columns[:custom_data_transformed.shape[1]]
    custom_data = pd.DataFrame(custom_data_transformed, columns=columns_after_imputation, index=j_data.index)

    if len(new_selected_features) > 0:
        return custom_data[new_selected_features]
    else:
        preprocessed_data = pipeline.named_steps['preprocessor'].transform(x)
        feature_names = get_feature_names(pipeline.named_steps['preprocessor'])
        ohe_data = pd.DataFrame(preprocessed_data, columns=feature_names, index=x.index)
    return ohe_data


# #############################################
# ######### Permutation Importance ############
# #############################################


try:
    import eli5
    from eli5.sklearn import PermutationImportance

    eli5_data = shap_and_eli5_custom_format(X_test)
    permuter = PermutationImportance(pipeline.named_steps['RandomForestRegressor'],
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
    print("This model does not support ELI5 library.")
