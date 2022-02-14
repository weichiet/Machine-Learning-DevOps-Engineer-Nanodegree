import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion
def merge_multiple_dataframe():

    #Get all the files name and select csv files only
    all_files = os.listdir(os.getcwd() + '/' + input_folder_path)
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))

    input_folder = os.getcwd() + '/' + input_folder_path + '/'
    output_folder = os.getcwd() + '/' + output_folder_path + '/'

    if csv_files:
        #Read the first file
        df_list = pd.read_csv(input_folder + csv_files[0])

        # Combine all files
        for each_filename in csv_files[1:]:
            df = pd.read_csv(input_folder + each_filename)
            df_list = df_list.append(df)

    # Create the 'output_folder_path' if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Drop the duplicats and save the final result
    result = df_list.drop_duplicates()
    result.to_csv(output_folder + 'finaldata.csv', index=False)

    # Save ingetsion location
    with open(output_folder + 'ingestedfiles.txt', 'w') as f:
        for each_filename in csv_files:
            f.write(each_filename + '\n')

if __name__ == '__main__':
    merge_multiple_dataframe()
