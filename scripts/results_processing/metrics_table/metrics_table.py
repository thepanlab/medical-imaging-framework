from sklearn.metrics import balanced_accuracy_score as weighted_acc
from sklearn.metrics import accuracy_score as unweighted_acc
from sklearn.metrics import f1_score as f1
from util.get_config import parse_json
from scipy.stats import sem as std_err
from termcolor import colored
from util import path_getter
from statistics import mean
import pandas as pd
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
    accs = {'individual': {}, 'column': {}, 'row': {}}
    for config in pred:
        accs['individual'][config] = {}
        accs['column'][config] = {}
        accs['row'][config] = {}
        
        # Get the accuracy and other metrics of every test-val fold pair
        for test_fold in pred[config]:
            accs['individual'][config][test_fold] = {'unweighted': {}, 'weighted': {}, 'f1': {}}
            accs['row'][config][test_fold] = {'unweighted': [], 'weighted': [], 'f1': []}

            # Get the accuracy
            for val_fold in pred[config][test_fold]:
                pred_vals = pred[config][test_fold][val_fold]
                true_vals = true[config][test_fold][val_fold]
                
                # Test-val pair
                accs['individual'][config][test_fold]['unweighted'][val_fold] = unweighted_acc(y_true=true_vals, y_pred=pred_vals)
                accs['individual'][config][test_fold]['weighted'][val_fold] = weighted_acc(y_true=true_vals, y_pred=pred_vals)
                accs['individual'][config][test_fold]['f1'][val_fold] = f1(y_true=true_vals, y_pred=pred_vals, average='weighted')
                
                # The row accuracy avg
                accs['row'][config][test_fold]['unweighted'] += [accs['individual'][config][test_fold]['unweighted'][val_fold]]
                accs['row'][config][test_fold]['weighted'] += [accs['individual'][config][test_fold]['weighted'][val_fold]]
                accs['row'][config][test_fold]['f1'] += [accs['individual'][config][test_fold]['f1'][val_fold]]
                
                # The column accuracy avg
                if val_fold not in accs['column'][config]:
                    accs['column'][config][val_fold] = {'unweighted': [], 'weighted': [], 'f1': []}
                accs['column'][config][val_fold]['unweighted'] += [accs['individual'][config][test_fold]['unweighted'][val_fold]]
                accs['column'][config][val_fold]['weighted'] += [accs['individual'][config][test_fold]['weighted'][val_fold]]
                accs['column'][config][val_fold]['f1'] += [accs['individual'][config][test_fold]['f1'][val_fold]]
            
            
            # Get the average test fold (row) accuracy
            accs['row'][config][test_fold]['unweighted'] = mean(accs['row'][config][test_fold]['unweighted'])
            accs['row'][config][test_fold]['weighted'] = mean(accs['row'][config][test_fold]['weighted'])
            accs['row'][config][test_fold]['f1'] = mean(accs['row'][config][test_fold]['f1'])
                
        # Get the average val fold (col) accuracies
        for val_fold in accs['column'][config]:
            accs['column'][config][val_fold]['unweighted'] = mean(accs['column'][config][val_fold]['unweighted'])
            accs['column'][config][val_fold]['weighted'] = mean(accs['column'][config][val_fold]['weighted'])
            accs['column'][config][val_fold]['f1'] = mean(accs['column'][config][val_fold]['f1'])
            
    # Calculate the standard error of the accuracies
    errs = {}
    for config in pred:
        errs[config] = {}
        
        # Get the standard error of the test-fold accs
        for test_fold in pred[config]:
            errs[config][test_fold] = {
                'unweighted': std_err(list(accs['individual'][config][test_fold]['unweighted'].values())),
                'weighted': std_err(list(accs['individual'][config][test_fold]['weighted'].values())),
                'f1': std_err(list(accs['individual'][config][test_fold]['f1'].values()))
            }
                
    # Return accuracy
    return accs, errs


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
    configs = list(accuracies['individual'].keys())
    test_folds = list(accuracies['individual'][configs[0]].keys())
    val_folds = list(standard_error[configs[0]].keys())
    test_folds.sort()
    val_folds.sort()
        
    # The metrics to write out to
    metrics = ['unweighted', 'weighted', 'f1']
        
    # Create a table for each config
    for config in configs:
        for metric in metrics:
            df = pd.DataFrame(columns=val_folds+['avg_acc', 'stderr'])
            
            # For every test fold, add column values
            for test_fold in test_folds:
                row = {}          
                
                # Add the individual accuracies
                for val_fold in accuracies['individual'][config][test_fold][metric]:
                    row[val_fold] = [accuracies['individual'][config][test_fold][metric][val_fold]]
                    
                # Add the mean and stderr
                row['avg_acc'] = [accuracies['row'][config][test_fold][metric]]
                row['stderr'] = [standard_error[config][test_fold][metric]]
                
                # Append the row to the dataframe
                df.loc[test_fold] = pd.DataFrame.from_dict(row).loc[0]
                
            # Add a row of val-fold accuracies
            row = {}
            for val_fold in val_folds:
                row[val_fold] = [accuracies['column'][config][val_fold][metric]]
            df.loc['avg_acc'] = pd.DataFrame.from_dict(row).loc[0]
                
            # Create and save the Pandas dataframe
            config_path = os.path.join(output_path, config)
            if not os.path.exists(config_path): os.makedirs(config_path)
            df.to_csv(f'{config_path}/{output_file}_{config}_{metric}.csv')


def main(config=None):
    """ The main program.

    Args:
        config (dict): A JSON configuration as a dictionary. (Optional)
    """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('./results_processing/metrics_table/metrics_table_config.json')

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
    print(colored("Finished writing the metric tables.", 'green'))


if __name__ == "__main__":
    """ Executes the program """
    main()
