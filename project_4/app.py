from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
#import diagnosis
#import predict_exited_from_saved_model
import json
import os

from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, missing_values, execution_time, outdated_packages_list

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    #call the prediction function you created in Step 3
    input_data_path = request.args.get("input_data")
    df = pd.read_csv(input_data_path)
    y_pred = model_predictions(df)

    # Return value for prediction outputs
    return str(list(y_pred))

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():

    #Check the score of the deployed model
    f1_score = score_model(test_data_path, model_path)

    #Add return value (a single F1 score number)
    return str(f1_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    #check means, medians, and modes for each column
    summary_stat = dataframe_summary()

    #return a list of all calculated summary statistics
    return str(summary_stat)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    #check timing and percent NA values
    timing = execution_time()
    missing_value = missing_values()
    outdated_list = outdated_packages_list()

    return str(f'Execution time: {timing} \n Missing data(%): {missing_value} \n Outdated packages: {outdated_list}')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
