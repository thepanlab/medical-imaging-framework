import os
import math
import re
from pathlib import Path
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as weighted_acc
from sklearn.metrics import accuracy_score as unweighted_acc
from sklearn.metrics import f1_score as f1
from util.get_config import parse_json
from util.predicted_formatter.predicted_formatter import main as reformat
from util.predicted_formatter.predicted_formatter import translate_file
from util.predicted_formatter.predicted_formatter import write_file
from termcolor import colored
from util import path_getter


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
                try:
                    results[config][test_fold][val_fold] = pd.read_csv(validation_fold_path, header=None).to_numpy()
                except:
                    print(colored(f"Warning: {validation_fold_path} is empty.", 'yellow'))

    # Return the dictionary
    return results

def read_data_outer(l_paths):
    """ From a list of paths of files with indices, it gets 
    a dictionary where the key is the test fold and
    the value is a numpy array

    Args:
        l_paths (list): list of paths of files with indices

    Returns:
        dict: A dictionary with key: test fold and value: numpy array.
    """
    # Return this
    results = {}

    # example of filenames
    # resnet_50_0_test_fold8_label_index.csv
    # resnet_50_test_E1_test-true-label-index.csv
    for path in l_paths:
        m = re.search('(?<=test_)[a-zA-Z0-9-]+', path.stem)
        test_name = m.group(0)

        a_value_index = pd.read_csv(path, header=None).to_numpy()

        results[test_name] = a_value_index

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
                
                # sample standard deviation
                stdev = math.sqrt(stdev/(n_val_folds-1))
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

def get_list_true_and_prediction_outer(config):
    """ Based on the 'data_path' parameter in the config.
    It returns list of path of true value indices and 
    list of path of prediction indices
    Args:
        config (dict): A JSON configuration as a dictionary

    Returns:
        Two lists: 
        * list with paths containing true value indices.
        * list with paths containing prediction indices.

    """

    path_data = Path(config['data_path'])
    
    # Search for true_label files with index
    l_true_label_index = list(path_data.rglob("*/true_label/*test_*index*"))
    
    if len(l_true_label_index) == 0:
        raise RuntimeError(f"config['data_path'] doesn't contain folder(s) " + 
                            "with name true_label that contain files with index")
    
    # Search for prediction
    l_prediction = list(path_data.rglob("*/prediction/"))
    
    for prediction_folder in l_prediction:
        l_files = list(prediction_folder.glob("*"))
        if len(l_files) == 1:
            translated_data = translate_file(l_files[0])
            write_file(l_files[0].as_posix(), translated_data)
            print(colored(f"Creating index file for {prediction_folder}", "green"))            
            
    l_prediction_index = list(path_data.rglob("*/prediction/*test_*index*"))

    return l_true_label_index, l_prediction_index


def save_dataframe_outer(dict_true_index, dict_prediction_index, config):
    """ 
    Based on the dictionaries it calculates: accuracy, weighted accuracy
    and f1 weighted per test fold. Then, it creates a pandas dataframe 
    and saves it as csv in 'output_path' and 'output_filename' as prefix.
    
    Args:
        dict_true_index (dict): 
        dict_prediction_index (dict): 
        config (dict): A JSON configuration as a dictionary

    """
    l_test_folds = dict_true_index.keys()
        
    # TODO
    # calculate accuracy and other metrics
    dict_metrics = {'accuracy': unweighted_acc, 'weighted_accuracy': weighted_acc, 'f1_weighted': f1}
    
    df_all = pd.DataFrame()
    
    for test_fold in l_test_folds:
        dict_row = {"test_fold":[test_fold]}
        for metric in dict_metrics:
            
            true_vals = dict_true_index[test_fold]
            pred_vals = dict_prediction_index[test_fold]
            
            if metric == 'f1_weighted':
                metric_value = dict_metrics[metric](y_true=true_vals,
                                    y_pred=pred_vals, average='weighted')
            else:
                metric_value = dict_metrics[metric](y_true=true_vals,
                                                    y_pred=pred_vals)
                
            dict_row[metric] = metric_value

        df_temp = pd.DataFrame(dict_row)
        # join them in pandas dataframe 
        df_all = pd.concat([df_all, df_temp], ignore_index=True)

    p_path_folder_output = Path(config["output_path"])

    p_path_folder_output.mkdir(parents=True, exist_ok=True)
    
    prefix_output = config["output_filename"]
    
    name_file = f"{prefix_output}_outer.csv"
    
    path_fileoutput = p_path_folder_output / name_file

    df_all = df_all.sort_values("test_fold").reset_index(drop=True)
    df_all = df_all.set_index('test_fold')
    n_testfolds = df_all.shape[0]
    
    df_all.loc['mean'] = df_all.mean()
    df_all.loc['std_err'] = df_all.std()/n_testfolds**0.5

    df_all.to_csv(path_fileoutput)
    print(colored(f"summary results for outer saved in: {path_fileoutput}.", 'green'))
    # and save
    
def main(config=None):
    """ The main program.

    Args:
        config (dict): A JSON configuration as a dictionary. (Optional)
    """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('./results_processing/summary_table/summary_table_config.json')

    if config["is_outer"] == False:
        
        # Get the necessary input files
        true_paths = path_getter.get_subfolder_files(config['data_path'], "true_label", isIndex=True, getValidation=True, isOuter=config['is_outer'])
        pred_paths = path_getter.get_subfolder_files(config['data_path'], "prediction", isIndex=True, getValidation=True, isOuter=config['is_outer'])

        # Read in each file into a dictionary
        true = read_data(true_paths)
        pred = read_data(pred_paths)
        
        # Get accuracies and standard error
        accuracies, stderr = get_accuracies_and_stderr(true, pred)

        # Graph results
        total_output(accuracies, stderr, config['output_path'], config['output_filename'], config['round_to'], is_outer=config['is_outer'])
        print(colored("Finished writing the summary tables.", 'green'))
    
    # config["is_outer"] == False
    else:

        l_true_label_index, l_prediction_index = get_list_true_and_prediction_outer(config)

        dict_true_label_index = read_data_outer(l_true_label_index)
        dict_prediction_index= read_data_outer(l_prediction_index)

        save_dataframe_outer(dict_true_label_index, dict_prediction_index, config)
        

if __name__ == "__main__":
    """ Executes the program """
    main()
