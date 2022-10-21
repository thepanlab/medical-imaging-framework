from sklearn.metrics import balanced_accuracy_score as weighted_acc
from sklearn.metrics import accuracy_score as unweighted_acc
from sklearn.metrics import f1_score as f1
from util.get_config import parse_json
from termcolor import colored
from util import path_getter
import pandas as pd
import math
import os


def read_data(paths):
    """ This will read in file-data into a config-subject dictionary

    Args:
        paths (dict): A dictionary of paths sorted by config and test fold.

    Raises:
        Exception: If no data files are found within some config and test fold.

    Returns:
        dict: A dictionary of dataframes.
    """
    # Return this
    results = {}

    # Create a dictionary for each config
    for config in paths:
        results[config] = {}

        # Create an array for each target-id
        for test_fold in paths[config]:
            results[config][test_fold] = {}

            # Check if the fold has files to read from
            if not paths[config][test_fold]:
                raise Exception(colored("Error: config '" + config + "' and testing fold '"
                                        + test_fold + "' had no indexed CSV files detected. "
                                        + "Double-check the data files are there.", 'red'))

            # For each file, read and store it
            for validation_fold_path in paths[config][test_fold]:
                val_fold = validation_fold_path.split("/")[-3].split("_")[-1]
                results[config][test_fold][val_fold] = pd.read_csv(validation_fold_path, header=None).to_numpy()

    # Return the dictionary
    return results


def get_accuracies_and_stderr(true, pred):
    """ Gets the accuracies of each fold-config index.

    Args:
        true (dict): A dictionary of true values sorted by config and test fold.
        pred (dict): A dictionary of predicted values sorted by config and test fold.

    Returns:
        dict: Two dictionaries of relative and real accuracies and standard error. Sorted by config and test fold.
    """
    # Get a table of accuracy for every config
    acc_tables = {'weighted': {}, 'unweighted': {}, 'f1_weighted': {}}
    acc_functions = {'weighted': weighted_acc, 'unweighted': unweighted_acc, 'f1_weighted': f1}
    for key in acc_tables:
        for config in pred:
            acc_tables[key][config] = {}
            
            # Get the accuracy and other metrics of every test-val fold pair
            for test_fold in pred[config]:
                acc_tables[key][config][test_fold] = {}
                for val_fold in pred[config][test_fold]:
                    
                    # Get the metrics of the test-val pair
                    pred_vals = pred[config][test_fold][val_fold]
                    true_vals = true[config][test_fold][val_fold]
                    if key == 'f1_weighted':
                        acc_tables[key][config][test_fold][val_fold] = acc_functions[key](y_true=true_vals, y_pred=pred_vals, average='weighted')
                    else:
                        acc_tables[key][config][test_fold][val_fold] = acc_functions[key](y_true=true_vals, y_pred=pred_vals)
                    
    # Get the means of the absolute and relative accuracies
    mean_accs = {'weighted': {}, 'unweighted': {}, 'f1_weighted': {}}
    for key in mean_accs:
        for config in acc_tables[key]:
            mean_accs[key][config] = {}
            
            # For every test-fold, sum the val-fold accs
            for test_fold in acc_tables[key][config]:
                mean_accs[key][config][test_fold] = 0
                for val_fold in acc_tables[key][config][test_fold]:
                    mean_accs[key][config][test_fold] += acc_tables[key][config][test_fold][val_fold]
                    
                # Divide by the number of summed val folds
                n_val_folds = len(acc_tables[key][config][test_fold])
                mean_accs[key][config][test_fold] /= n_val_folds
                   
    # Calculate the standard error of the means
    mean_errs = {'weighted': {}, 'unweighted': {}, 'f1_weighted': {}}
    for key in mean_errs:
        for config in acc_tables[key]:
            mean_errs[key][config] = {}
            
            # For every test-fold, sum the val-fold accs and divide by n val folds
            for test_fold in acc_tables[key][config]:
                stdev = 0
                
                # Sum the differences squared of the accuracy and its mean
                for val_fold in acc_tables[key][config][test_fold]:                    
                    stdev += (acc_tables[key][config][test_fold][val_fold] - mean_accs[key][config][test_fold])**2
                
                # Get the standard deviation and the mean's error
                n_val_folds = len(acc_tables[key][config][test_fold])
                mean_errs[key][config][test_fold] = stdev / math.sqrt(n_val_folds)
                
    # Return accuracy
    return mean_accs, mean_errs


def total_output(accuracies, standard_error, output_path, output_file, round_to, is_outer):
    """ Produces a table of accuracies and standard errors by config and fold

    Args:
        accuracies (dict): A dictionary of accuracies sorted by config and test fold.
        standard_error (dict): A dictionary of errors sorted by config and test fold.
        output_path (str): A string of the directory the output CSV should be written to.
        output_file (str): Name prefix of the output files.
        is_outer (bool): If the data is from the outer loop.
    """
    # Get names of columns and subjects
    config_names = list(accuracies['unweighted'].keys())
    test_folds = list(accuracies['unweighted'][config_names[0]].keys())
    
    # Alter output path
    if is_outer:
        output_path = os.path.join(output_path, 'outer_loop')
    else:
        output_path = os.path.join(output_path, 'inner_loop')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Make a file for each dataframe
    dfs = {
        "acc": accuracies['unweighted'], 
        "acc_weighted": accuracies['weighted'], 
        
        "acc_err": standard_error['unweighted'],
        "acc_err_weighted": standard_error['weighted'],
        
        "f1_weighted": accuracies['f1_weighted'],
        "f1_err_weighted": standard_error['f1_weighted']
    }
    for key in dfs:
        
        # Create the dataframe's arrays by column (or config)
        data = {}
        for config in config_names:
            data[config] = []
            data[config] = []
            for test_fold in test_folds:
                data[config].append(dfs[key][config][test_fold])

        # Create and save the Pandas dataframe
        df = pd.DataFrame(data=data, index=test_folds)
        df.index.names = ['test_fold']
        df = df.sort_values(by=['test_fold'], ascending=True)
        if is_outer:
            df.round(round_to).to_csv(f'{output_path}/{output_file}_outer_{key}.csv')
        else:
            df.round(round_to).to_csv(f'{output_path}/{output_file}_inner_{key}.csv')


def main(config=None):
    """ The main program.

    Args:
        config (dict): A JSON configuration as a dictionary. (Optional)
    """
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
    total_output(accuracies, stderr, config['output_path'], config['output_filename'], config['round_to'], is_outer=False)
    print(colored("Finished writing the summary tables.", 'green'))


if __name__ == "__main__":
    """ Executes the program """
    main()
