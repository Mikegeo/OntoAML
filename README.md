# OntoAML

Pre-requisites:
1) Pycharm or any other python IDE and python 3.9
2) git clone the repo
3) pip install -r requirements.txt


Before running the main.py open the /create/create_main.py 
1) at line 32 change the path to the path of your java.exe (preferable download Protege ontology editor and use that
Java.exe) 

2) at line 555 rename the path again suitable to your project path.

# Running the system phase 1 training:
run main_page.py through the IDE

1) Select create pipelines
2) Upload the 'csv' only file
3) Press Analyse Dataset
4) Select if you want to save the new folder to the Desktop or another path
5) View Dataset Profile (optional)
6) Select the Models type and the hyperparameter tuning method
7) Select the target column (y)
8) Select features that you would like to remove from the training session (optional)
9) Press Create Machine learning pipeline
10) The new Python-based machine learning pipelines will be in the folder path you created at the dataset analysis
11) Manually run the Modelname_pipeline.py 
12) The model and the evaluation metrics will be saved at the same folder with the 'Modelname'_pipeline.py 

# Prediction Phase 
run main_page.py through the IDE. (Not fully tested may introduce errors on some experiments.)

1) Press Make Predictions
2) Upload the whole experiment folder with the trained pipelines.
3) Upload the test dataset (must be the same number of csv columns as the train set, except the target column).
4) Show best model evaluation metrics (this shows the best ML pipeline metrics)
5) Show best pipeline explanation (this shows the best ML pipeline explanations)
6) To make explainable predictions you need to choose a row from the test dataset you uploaded, just the row number (e.g. 2)
7) Then select the three explaindability methods to make explainable predictions.

# If prediction phase does not work
If the explainable predictions are not working you can create a script and use the trained model to make manually the predictions there.


# Utilities for a timed faster training if you choose grid search
Open utils/hp_tuning/

Replace the grid search hp tuning method (inside the Python-based pipelines) with the corresponding in this folder.
Select your preferred time budget, measured in seconds. (time_budget = 60 -> 1 min of training.)
