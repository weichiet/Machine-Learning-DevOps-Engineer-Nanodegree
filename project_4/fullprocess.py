import os
import sys
import json
import subprocess
import pandas as pd

from ingestion import merge_multiple_dataframe
from scoring import score_model
from training import train_model

with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_folder_path = config['output_folder_path']

##################Check and read new data
#first, read ingestedfiles.txt
with open(os.path.join(prod_deployment_path, "ingestedfiles.txt"), "r") as f:
    ingested_files = [line.rstrip() for line in f]

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
input_files = os.listdir(input_folder_path)

have_new_files = False
for file_name in input_files:
    if file_name not in ingested_files:
        have_new_files = True

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not have_new_files:
    print('No new files were found')
    sys.exit()

##################Checking for model drift
#Check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

# Read the score from the latest model, recorded in latestscore.txt
with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as f:
    old_score = float(f.read())

# Merge the new data as a single dataframe
merge_multiple_dataframe()
# Make predictions using the trainedmodel.pkl model in the production_deployment with new mergerd data
new_score = score_model(output_folder_path, prod_deployment_path)

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
print(f'New F1 score:{new_score:.3f} v.s Old F1 score:{old_score:.3f}')
if new_score >= old_score:
    print('New F1 score does not indicate model drift')
    sys.exit()

print('New F1 score indicates model drift. Starting model retraining..')
train_model()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
subprocess.run(['python', 'deployment.py'])

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
subprocess.run(['python', 'reporting.py'])

subprocess.run(['python', 'apicalls.py'])
