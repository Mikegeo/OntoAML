# Grid search implementation for regression
import time
from itertools import product
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import json
n = 4
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
best_hyperparameters = os.path.join(file_dir, 'best_hyperparameters.json')
with open(best_hyperparameters, 'w') as file:
    json.dump(best_params_dict, file, indent=4)
###########################################################################
###########################################################################