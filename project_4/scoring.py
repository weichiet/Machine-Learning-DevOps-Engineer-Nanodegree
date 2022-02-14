from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])

def get_file_full_path(directory, file_type='.csv'):
    '''
    Given the directory, this function find the speficy file type in the file directory
    and return the full path of the file.
    It assume there is only one single file with the specify file type in the directory.
    '''

    files_directory = os.getcwd() + '/' + directory + '/'
    all_files = os.listdir(files_directory)
    file_name = list(filter(lambda f: f.endswith(file_type), all_files))

    return files_directory + file_name[0]

#################Function for model scoring
def score_model(data_path, model_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    #Read the test datast
    test_data_full_path = get_file_full_path(data_path, '.csv')
    test_data = pd.read_csv(test_data_full_path)

    # Load the trained ML model
    model_full_path = get_file_full_path(model_path, '.pkl')
    with open(model_full_path, 'rb') as file:
        model = pickle.load(file)

    # Split test data into X and y
    X = test_data[['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)
    y = test_data['exited'].values.reshape(-1,1)

    # Make inference
    predicted = model.predict(X)
    #Calculate F1 score
    f1_score = metrics.f1_score(predicted,y)

    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)

    # Save score
    with open(os.getcwd() + '/' + output_model_path + '/' + 'latestscore.txt', 'w') as f:
        f.write(str(f1_score))

    return f1_score

if __name__ == "__main__":
    print(score_model(test_data_path, output_model_path))
