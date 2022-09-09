import summary_table as filereader
from get_config import parse_json
from termcolor import colored
import pandas as pd
import numpy as np
import path_getter
import math
import os



def count_epochs(history_paths):
    """ Creates a csv file for every model's history """
    # Store output dataframes by model
    model_dfs = {}

    # Each model will have its own dictionary, a later dataframe
    for model in history_paths:
        model_dfs[model] = {}

        # Every subject is the kth test set (row), find the column values
        col_names = []
        for row in history_paths[model]:
            row_name = row
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
                model_dfs[model][row_name][col_name] = epochs
    
    # Return a dictionary of counts
    return model_dfs



def print_counts(epochs, output_path):
    """ This will output a CSV of the epoch-counts """
    # Create a new dataframe to output
    col_names = ["test_fold", "config", "val_fold", "epochs"]
    df = pd.DataFrame(columns=col_names)

    # Re-format data to match the columns above
    for config in epochs:
        for testing_fold in epochs[config]:
            for validation_fold in epochs[config][testing_fold]:
                
                # Each row should contain the given columns
                df = df.append({
                    col_names[0]: testing_fold, 
                    col_names[1]: config, 
                    col_names[2]: validation_fold, 
                    col_names[3]: epochs[config][testing_fold][validation_fold]
                }, ignore_index=True)

    # Print to file
    file_name = 'epochs.csv'
    df.to_csv(os.path.join(output_path, file_name), index=False)
    print(colored('Successfully printed epoch results to: ' + file_name, 'green'))



def print_stderr(epochs, data_path, output_path):
    """ This will output a CSV of the average epoch standard errors """
    # Get the necessary input files
    true_paths, pred_paths = filereader.get_paths(data_path)

    # Read in each file into a dictionary
    true = filereader.read_data(true_paths)
    pred = filereader.read_data(pred_paths)

    # Get accuracies and standard error
    accuracies, stderr = filereader.get_accuracies_and_stderr(true, pred)

    # Create a new dataframe to output
    col_names = ["test_fold", "config", "avg_epochs", "std_err"]
    df = pd.DataFrame(columns=col_names)

    # Re-format data to match the columns above
    for config in epochs:
        for test_fold in epochs[config]:

            # Count epochs
            epoch_total = 0
            for validation_fold in epochs[config][test_fold]:
                epoch_total += epochs[config][test_fold][validation_fold]

                
            # Each row should contain the given columns
            df = df.append({
                col_names[0]: test_fold, 
                col_names[1]: config, 
                col_names[2]: epoch_total / len(epochs[config][test_fold]), 
                col_names[3]: stderr[config][test_fold]
            }, ignore_index=True)

    # Print to file
    file_name = 'epoch_avg_and_stderr.csv'
    df.to_csv(os.path.join(output_path, file_name), index=False)
    print(colored('Successfully printed epoch averages/stderrs to: ' + file_name, 'green'))
                
            

def main(config=None):
    """ The main body of the program """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('epoch_counting_config.json')
    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])

    # Get the necessary input files
    history_paths = path_getter.get_history_paths(config['data_path'])

    # Count the number of epochs within every file
    epochs = count_epochs(history_paths)

    # Output the counts
    print_counts(epochs, config['output_path'])

    # Output the stderr
    print_stderr(epochs, config['data_path'], config['output_path'])
    



if __name__ == "__main__":
    main()
