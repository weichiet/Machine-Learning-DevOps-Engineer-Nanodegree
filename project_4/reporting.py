import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import ConfusionMatrixDisplay
from diagnostics import model_predictions

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

##############Function for reporting
def score_model(test_data):
    #Calculate a confusion matrix using the test data and the deployed model

    y_test = test_data["exited"].values.reshape(-1, 1).ravel()

    y_pred = model_predictions(test_data)

    #Write the confusion matrix to the workspace
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig(os.path.join(output_model_path, "confusionmatrix.png"))

if __name__ == '__main__':
    score_model(test_data)
