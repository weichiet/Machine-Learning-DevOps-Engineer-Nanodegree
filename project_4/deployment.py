from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])
output_folder_path = config['output_folder_path']
prod_deployment_path = os.path.join(config['prod_deployment_path'])

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    #Create the 'output_folder_path' if it doesn't exist
    output_folder = os.getcwd() + '/' + prod_deployment_path + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_name = 'trainedmodel.pkl'
    score_name = 'latestscore.txt'
    ingested_record_name = 'ingestedfiles.txt'

    # Copy the files
    os.system('cp' + ' ' + model_path + '/' + model_name + ' ' + prod_deployment_path + '/' + model_name)
    os.system('cp' + ' ' + model_path + '/' + score_name + ' ' + prod_deployment_path + '/' + score_name)
    os.system('cp' + ' ' + output_folder_path + '/' + ingested_record_name + ' ' + prod_deployment_path + '/' + ingested_record_name)


if __name__ == "__main__":
    store_model_into_pickle()
