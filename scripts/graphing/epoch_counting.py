from get_config import parse_json
from termcolor import colored
import pandas as pd
import numpy as np
import path_getter
import math
import os



def output_csv(history_paths, output_path):
    """ Creates a csv file for every model's history """
    # Store output dataframes by model
    model_dfs = {}

    # Each model will have its own dictionary, a later dataframe
    for model in history_paths:
        model_dfs[model] = {}

        # Every subject is the kth test set (row), find the column values
        col_names = []
        for row in history_paths[model]:
            row_name = 'test_' + row
            model_dfs[model][row_name] = {}


            # Read in the data at this path
            for path in history_paths[model][row]:
                # Check if the fold has files to read from
                if not path:
                    raise Exception(colored("Error: model '" + model + "' and target '" + target_id 
                                                        + "' had no history file detected.", 'red'))

                # Read the file for this column/subject, get number of rows (epochs)
                data = pd.read_csv(path)
                epochs = data.shape[0]

                # Add to the model's dataframe 
                col_name = path.split('/')[-2][-2:]
                if col_name not in col_names:
                    col_names.append(col_name)
                model_dfs[model][row_name][col_name] = epochs

        # Re-structure the dictionary to easy conversion
        col_names.sort()
        to_df = {}
        for row in model_dfs[model]:
            to_df[row] = [0] * int(col_names[-1][-1])
            for col in model_dfs[model][row]:
                to_df[row][int(col[-1])-1] = model_dfs[model][row][col]

        # Convert the dictionary to a dataframe
        model_dfs[model] = pd.DataFrame.from_dict(to_df, orient='index')
        model_dfs[model].columns = ['val_e' + str(col) for col in model_dfs[model].columns]

        # Print the dataframe to file
        file_name = model + '_epochs.csv'
        model_dfs[model].to_csv(os.path.join(output_path, file_name))
        print(colored('Successfully printed ' + model + "'s epoch results to file.", 'green'))

                
            

def main(config=None):
    """ The main body of the program """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('epoch_counting_config.json')
    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])

    # Get the necessary input files
    history_paths = path_getter.get_history_paths(config['data_path'])
    output_csv(history_paths, os.path.join(config['output_path']))


if __name__ == "__main__":
    main()
