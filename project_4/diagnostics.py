
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

##################Function to get model predictions
def model_predictions(test_data):
    #read the deployed model and a test dataset, calculate predictions
    model_full_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(model_full_path, 'rb') as file:
        model = pickle.load(file)

    X_test = test_data.loc[:, ["lastmonth_activity", "lastyear_activity",
                           "number_of_employees"]].values.reshape(-1, 3)
    y_test = test_data["exited"].values.reshape(-1, 1).ravel()

    predicted = model.predict(X_test)

    #Return value should be a list containing all predictions
    return predicted

##################Function to get summary statistics
def dataframe_summary():

    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    numeric_columns = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"
    ]

    summary_statistics = []
    for column in numeric_columns:
        summary_statistics.append([column, "mean", df[column].mean()])
        summary_statistics.append([column, "median", df[column].median()])
        summary_statistics.append(
            [column, "standard deviation", df[column].std()])

    return summary_statistics

##################Function to check for missing data
def missing_values():
    # calculate percentage of the missing values by columns
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    nas = list(df.isna().sum())
    napercents = [nas[i]/len(df.index) for i in range(len(nas))]

    return napercents

##################Function to get timings
def execution_time():
    #Calculate timing of training.py and ingestion.py
    all_timings = []
    for task in ["training.py", "ingestion.py"]:
        starttime = timeit.default_timer()
        os.system(f'python3 {task}')
        timing = timeit.default_timer() - starttime
        all_timings.append([task, timing])

    #Return a list of 2 timing values in seconds
    return all_timings

##################Function to check dependencies
def outdated_packages_list():
    #get a list of outdated packages
    outdated_packages = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return outdated_packages

if __name__ == '__main__':
    print(model_predictions(test_data))
    print(dataframe_summary())
    print(execution_time())
    print(outdated_packages_list())
