from get_config import parse_json
from termcolor import colored
import pandas as pd
import numpy as np
import path_getter
import math
import os



def read_data(paths):
    """ This will read in file-data into a config-subject dictionary """
    # Return this
    results = {}

    # Create a dictionary for each config
    for config in paths:
        results[config] = {}

        # Create an array for each target-id
        for target_id in paths[config]:
            results[config][target_id] = {}

            # Check if the fold has files to read from
            if not paths[config][target_id]:
                raise Exception(colored("Error: config '" + config + "' and target '" 
                    + target_id +  "' had no indexed CSV files detected. " 
                    + "Double-check the predictions have files ending in '_index.csv'.", 'red'))

            # For each file, read and store it
            for path in paths[config][target_id]:
                val_fold_name = path.split("/")[-3][-2:]
                results[config][target_id][val_fold_name] = pd.read_csv(path, header=None).to_numpy()

    # Return the dictionary
    return results



def get_accuracies_and_stderr(true, pred):
    """ Gets the accuracies of each fold-config index """
    # Compute a dictionary of correctness
    totals = {}

    # For each config and fold, compare each result-index
    for config in pred:
        totals[config] = {}
        for test_fold in pred[config]:
            totals[config][test_fold] = {}

            # For each config's test subject's validation fold, count the correctness
            for val_fold in pred[config][test_fold]:
                totals[config][test_fold][val_fold] = {'correct': 0, 'total': 0, 'accuracy': 0}

                # Get the values
                predicted_values = pred[config][test_fold][val_fold]
                true_values = true[config][test_fold][val_fold]

                # Check whether each prediction matches the truth
                for i in range(len(predicted_values)):
                    totals[config][test_fold][val_fold]['total'] += 1
                    if predicted_values[i] == true_values[i]:
                        totals[config][test_fold][val_fold]['correct'] += 1
                
                # Find the accuracy of the validation fold
                totals[config][test_fold][val_fold]['accuracy'] = \
                    totals[config][test_fold][val_fold]['correct'] / totals[config][test_fold][val_fold]['total']

    # Get accuracies and standard error for each config and test fold
    accs = {}
    errs = {}
    for config in totals:
        accs[config] = {}
        errs[config] = {}
        for test_fold in totals[config]:

            # Combine the accuracies, totals, and accuracies for each validation fold
            accuracies = 0
            correct = 0
            total = 0
            for val_fold in totals[config][test_fold]:
                accuracies += totals[config][test_fold][val_fold]['accuracy']
                correct += totals[config][test_fold][val_fold]['correct']
                total += totals[config][test_fold][val_fold]['total']

            # Get the total test fold accuracy
            accs[config][test_fold] = correct / total
            
            # Get the mean accuracy
            n_val_folds = len(totals[config][test_fold])
            mean_acc = accuracies / n_val_folds
            
            # Calculate standard deviation = sqrt(sum((acc_i-mean)^2)/N)
            stdev = 0
            for val_fold in totals[config][test_fold]:
                stdev += (totals[config][test_fold][val_fold]['accuracy'] - mean_acc)**2
            stdev = math.sqrt(stdev / n_val_folds)

            # Get standard error = standard deviation/sqrt(N)
            errs[config][test_fold] = stdev / math.sqrt(n_val_folds)
    
    # Return accuracy
    return accs, errs



def output(accuracies, standard_error, output_path):
    """ Produces a table of accuracies and standard errors by config and fold """
    # Get names of columns and subjects
    config_names = list(accuracies.keys())
    test_folds = list(accuracies[config_names[0]].keys())

    # Get column and row labels for table
    col_labels = []
    row_labels = test_folds
    for name in config_names:
        col_labels.append(name + '_acc')
        col_labels.append(name + '_err')

    # Create the to-be-dataframe arrays by column (or config)
    data = {}
    for config in config_names:
        data[config + '_acc'] = []
        data[config + '_err'] = []
        for test_fold in test_folds:
            data[config + '_acc'].append(accuracies[config][test_fold])
            data[config + '_err'].append(standard_error[config][test_fold])

    # Create and save the Pandas dataframe
    df = pd.DataFrame(data=data, index=row_labels)
    df.index.names = ['test_fold']
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
    output(accuracies, stderr, os.path.join(config['output_path'], config['output_filename']))
    print(colored("Finished writing the summary table.", 'green'))



if __name__ == "__main__":
    """ Executes the program """
    main()
