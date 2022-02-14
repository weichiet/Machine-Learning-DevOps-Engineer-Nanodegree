from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])

#################Function for training the model
def train_model():

    # Read the training data
    trainingdata = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    # Split the data into X and y
    X = trainingdata.loc[:, ['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    y = trainingdata['exited'].values.reshape(-1, 1).ravel()

    #Use logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='ovr', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)


    #Fit the logistic regression to the data
    model = logit.fit( X, y)

    #Create the 'output_folder_path' if it doesn't exist
    output_folder = os.getcwd() + '/' + model_path + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #Write the trained model to the workspace in a file called trainedmodel.pkl
    pickle.dump(model, open((output_folder + 'trainedmodel.pkl'), 'wb'))

if __name__ == '__main__':
    train_model()
