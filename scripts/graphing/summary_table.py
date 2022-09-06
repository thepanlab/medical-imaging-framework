from get_config import parse_json
from termcolor import colored
import pandas as pd
import numpy as np
import path_getter
import math
import os



def read_data(paths):
    """ This will read in file-data into a model-subject dictionary """
    # Return this
    results = {}

    # Create a dictionary for each model
    for model in paths:
        results[model] = {}

        # Create an array for each target-id
        for target_id in paths[model]:
            results[model][target_id] = []

            # Check if the fold has files to read from
            if not paths[model][target_id]:
                raise Exception(colored("Error: model '" + model + "' and target '" 
                    + target_id +  "' had no indexed CSV files detected. " 
                    + "Double-check the predictions have files ending in '_index.csv'.", 'red'))

            # For each file, read and store it
            for path in paths[model][target_id]:
                data = pd.read_csv(path, header=None).to_numpy()
                results[model][target_id] = data

    # Return the dictionary
    return results



def get_accuracies_and_stderr(true, pred):
    """ Gets the accuracies of each fold-model index """
    # Compute a dictionary of correctness and of combined arrays
    totals = {}
    arrays = {}

    # For each model and fold, compare each result-index
    for model in pred:
        totals[model] = {}
        arrays[model] = {}
        for subject in pred[model]:

            # Will count the correct and total counts of all items
            totals[model][subject] = {'correct': 0, 'total': 0}
            arrays[model][subject] = {'pred': [], 'true': []}
            for index in range(len(pred[model][subject])):

                # Get the values
                predicted_values = pred[model][subject][index]
                true_values = true[model][subject][index]

                # Append values to array-lists
                arrays[model][subject]['pred'].extend(predicted_values)
                arrays[model][subject]['true'].extend(true_values)

                # Check whether each prediction matches the truth
                for i in range(len(predicted_values)):
                    totals[model][subject]['total'] += 1
                    if predicted_values[i] == true_values[i]:
                        totals[model][subject]['correct'] += 1

    # Get accuracies and standard error for each model and fold
    accs = {}
    errs = {}
    for model in pred:
        accs[model] = {}
        errs[model] = {}
        for subject in pred[model]:

            # Each model-fold has an accuracy (correct / total) and standard error (standard deviation / sqrt n)
            accs[model][subject] = totals[model][subject]['correct'] / totals[model][subject]['total']
            errs[model][subject] = np.std(
                [arrays[model][subject]['pred'],
                arrays[model][subject]['true']]) / math.sqrt(len(arrays[model][subject]['pred'])
            )
    
    # Return accuracy
    return accs, errs



def graph(accuracies, standard_error, output_path):
    """ Produces a table of accuracies and standard errors by model and fold """
    # Get names of columns and subjects
    model_names = list(accuracies.keys())
    subject_names = list(accuracies[model_names[0]].keys())

    # Get column and row labels for table
    col_labels = []
    row_labels = subject_names
    for name in model_names:
        col_labels.append(name + '_acc')
        col_labels.append(name + '_err')

    # Create the to-be-dataframe arrays by column (or model)
    data = {}
    for model in model_names:
        data[model + '_acc'] = []
        data[model + '_err'] = []
        for subject in subject_names:
            data[model + '_acc'].append(accuracies[model][subject])
            data[model + '_err'].append(standard_error[model][subject])

    # Create and save the Pandas dataframe
    df = pd.DataFrame(data=data, index=row_labels)
    df.to_csv(output_path + '.csv')



def main(config=None):
    """ The main body of the program """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('summary_table_config.json')

    # Get the necessary input files
    true_paths = path_getter.get_subfolder_files(config['data_path'], "true_label", isIndex=True, getValidation=True)
    pred_paths = path_getter.get_subfolder_files(config['data_path'], "prediction", isIndex=True, getValidation=True)

    # Read in each file into a dictionary
    true = read_data(true_paths)
    pred = read_data(pred_paths)

    # Get accuracies and standard error
    accuracies, stderr = get_accuracies_and_stderr(true, pred)

    # Graph results
    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])
    graph(accuracies, stderr, os.path.join(config['output_path'], config['output_filename']))
    print(colored("Finished writing the summary table.", 'green'))



if __name__ == "__main__":
    """ Executes the program """
    main()
