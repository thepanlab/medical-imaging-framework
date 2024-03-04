import os
import math
import re
from pathlib import Path
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from tqdm import tqdm
from util.get_config import parse_json
from util.predicted_formatter.predicted_formatter import main as reformat
from util.predicted_formatter.predicted_formatter import translate_file
from util.predicted_formatter.predicted_formatter import write_file
from termcolor import colored
from util import path_getter

def extract_test_val(path):
    """ From absolute path, it extracts the test, val and 
    the indices in a array

    Args:
        paths (pathlib.Path): Path to be extracted

    Returns:
        string: test name
        string: validation name
        numpy array: index values
    """
    m = re.search('(?<=test_)[a-zA-Z0-9-]+_val_[a-zA-Z0-9-]+', path.stem)
    re_text = m.group(0)

    l_split_text = re_text.split("_")

    test_name = l_split_text[0]
    val_name = l_split_text[2]

    a_value_index = pd.read_csv(path, header=None).to_numpy()

    return test_name, val_name, a_value_index


def process_list_true_and_predictions_inner(path_data):
    """ Based on the path_data. It returns list of path of true label indices and 
    prediction indices. To search it uses patter for inner files.
    Args:
        path_data (pahtlib.Path): parent folder to search values
    Returns:
        Two lists: 
        * list with paths containing true value indices.
        * list with paths containing prediction indices.
    """
    # Search for true_label files with index
    l_true_label_index = list(path_data.rglob("*/true_label/*test_*val*val*index*"))

    if len(l_true_label_index) == 0:
        raise RuntimeError(f"`path_data` doesn't contain folder(s) " + 
                        "with name true_label that contain files with index")

    l_prediction = list(path_data.rglob("*/prediction/*val*"))

    # get list of validation folds
    l_test_val_fold_names = []
    
    for file_prediction in l_prediction:
        m = re.search('(?<=test_)[a-zA-Z0-9-]+_val_[a-zA-Z0-9-]+', file_prediction.stem)
        combination_name = m.group(0)

        test_name = combination_name.split("_")[0]
        val_name = combination_name.split("_")[2]

        tuple_test_val = (test_name, val_name)

        if val_name not in l_test_val_fold_names:
            l_test_val_fold_names.append(tuple_test_val)


    for test_name, val_name in l_test_val_fold_names:
        l_prediction_val = list(path_data.rglob(f"*/prediction/*test_{test_name}_val_{val_name}*val*"))

        if len(l_prediction_val) == 1:
            translated_data = translate_file(l_prediction_val[0])
            write_file(l_prediction_val[0].as_posix(), translated_data)
            print(colored(f"Creating index file for {l_prediction_val[0]}", "green"))            
            
    l_prediction_index = list(path_data.rglob("*/prediction/*test_*val_*index*"))
    
    return l_true_label_index, l_prediction_index


def read_data_inner(l_paths, config):
    """ From a list of paths of files with indices, it gets:
    If a normal inner loop: 
    * a dictionary where the key is a tuple with the test fold and 
      validaiton name. The value is a numpy array

    a random search inner loop:
    * a dictionary where the key is a tuple with the random search name,
      test fold, validation name. The value is a numpy array

    
    Args:
        l_paths (list): list of paths of files with indices

    Returns:
        dict: A dictionary with value: numpy array.
    """
    # Return this
    results = {}

    if config["random_search_if_inner"] == False:

        l_results = Parallel(n_jobs=-1)(delayed(extract_test_val)(path) for path in l_paths)

        for test_name, val_name, a_value_index in l_results:
            results[(test_name,val_name)] = a_value_index
        
    
    # config["random_search_if_inner"] == True:
    else:

        # example of filenames
        # resnet_50_0_test_fold8_label_index.csv
        # resnet_50_test_E1_test-true-label-index.csv
        print(colored(f"Extracting paths from random search directories", "green"))  
        for rs_name, l_path_rs in tqdm(l_paths):
            
            l_results = Parallel(n_jobs=-1)(delayed(extract_test_val)(path) for path in l_path_rs)
        
            for test_name, val_name, a_value_index in l_results:
                results[(rs_name, test_name,val_name)] = a_value_index

    # Return the dictionary
    return results


def get_list_true_and_prediction(config):
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
    
    if config["is_outer"] == True:
        
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

    # config["is_outer"] == False
    else:
        if config["random_search_if_inner"] == False:

            l_true_label_index, l_prediction_index = process_list_true_and_predictions_inner(path_data)

        # config["random_search_if_inner"] == True:
        else:
            # get list of subfolds
            l_subdirectories = [x for x in path_data.iterdir() if x.is_dir()]

            l_true_label_index = []
            l_prediction_index = []
            
            print(colored(f"Extracting true labels and predictions indices from random search directories", "green"))  

            l_results = Parallel(n_jobs=-1)(delayed(process_list_true_and_predictions_inner)(subdirectory) for subdirectory in l_subdirectories)
            
            for result, subdirectory in zip(l_results, l_subdirectories):
            
                rs_name = subdirectory.stem

                l_true_label_index.append((rs_name, result[0]))
                l_prediction_index.append((rs_name, result[1]))

        if len(l_true_label_index) != len(l_prediction_index):
            raise RuntimeError(f"Length of true label index: {len(l_true_label_index)}" +
                            f" and length of predicion: {len(l_true_label_index)}" +
                            "are different.")

    return l_true_label_index, l_prediction_index

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

def save_dataframes(dict_true_index, dict_prediction_index, config):
    """ 
    Based on the dictionaries it calculates: accuracy, weighted accuracy
    and f1 weighted per test fold. Then, it creates a pandas dataframe 
    and saves it as csv in 'output_path' and 'output_filename' as prefix.
    
    Args:
        dict_true_index (dict): 
        dict_prediction_index (dict): 
        config (dict): A JSON configuration as a dictionary

    """
    
    if config["is_outer"] == False:
        
        l_test_val_folds = dict_prediction_index.keys()
        
        # calculate accuracy and other metrics
        dict_metrics = {'accuracy': accuracy_score, 'balanced_accuracy': balanced_accuracy_score,
                        'f1': f1_score}
                  
        df_all = pd.DataFrame()
  
        if config["random_search_if_inner"] == False:
            
            # TODO
            # FIX not output so far !!!! for inner loop
            for test_fold, val_fold in l_test_val_folds:
                dict_row = {"test_fold":[test_fold],
                            "val_fold":[val_fold]}
                for metric in dict_metrics:    
                                    
                    true_vals = dict_true_index[(test_fold, val_fold)]
                    pred_vals = dict_prediction_index[(test_fold, val_fold)]
                    
                    if metric == 'f1':
                        metric_value = dict_metrics[metric](y_true=true_vals,
                                            y_pred=pred_vals, average="weighted")
                    else:
                        metric_value = dict_metrics[metric](y_true=true_vals,
                                            y_pred=pred_vals)
                    
                    dict_row[metric] = metric_value
                        
                df_temp = pd.DataFrame(dict_row)
                # join them in pandas dataframe 
                df_all = pd.concat([df_all, df_temp], ignore_index=True)
                
            df_all=df_all.sort_values(by=['test_fold',"val_fold"]).reset_index(drop=True)
                
            df_test_mean = df_all.groupby(["test_fold"]).mean()
            df_test_stddev = df_all.groupby(["test_fold"]).std()
    
            series_count_test = df_all.groupby(["test_fold"]).count()["val_fold"]
            df_test_stderror = df_test_stddev.div(series_count_test**0.5, axis=0)
        
            p_path_folder_output = Path(config["output_path"])
            p_path_folder_output.mkdir(parents=True, exist_ok=True)
            
            prefix_output = config["prefix_filename"]
            
            name_file = f"{prefix_output}_all.csv"           
            path_fileoutput = p_path_folder_output / name_file
            
            df_all.to_csv(path_fileoutput)
            print(colored(f"Complete results for {metric} saved in: {path_fileoutput}.", 'green'))
            
            name_file = f"{prefix_output}_mean_per_test_fold.csv"
            path_fileoutput = p_path_folder_output / name_file

            df_test_mean.to_csv(path_fileoutput)
            print(colored(f"Mean results saved in: {path_fileoutput}.", 'green'))

            name_file = f"{prefix_output}_stderr_per_test_fold.csv"
            path_fileoutput = p_path_folder_output / name_file

            df_test_stderror.to_csv(path_fileoutput)
            print(colored(f"Standard error results saved in: {path_fileoutput}.", 'green'))
            
        # config["random_search_if_inner"] == True:
        else:
            for rs_name, test_fold, val_fold in l_test_val_folds:
                dict_row = {"rs_name":[rs_name],
                            "test_fold":[test_fold],
                            "val_fold":[val_fold]}
                                
                true_vals = dict_true_index[(rs_name, test_fold, val_fold)]
                pred_vals = dict_prediction_index[(rs_name, test_fold, val_fold)]
                
                for metric in dict_metrics:  
                    if metric == 'f1':
                        value = dict_metrics[metric](y_true=true_vals,
                                            y_pred=pred_vals, average="weighted")
                    else:
                        value = dict_metrics[metric](y_true=true_vals,
                                                     y_pred=pred_vals)
                                        
                    dict_row[metric] = value
                
                df_temp = pd.DataFrame(dict_row)
                # join them in pandas dataframe 
                df_all = pd.concat([df_all, df_temp], ignore_index=True)
            
            df_all=df_all.sort_values(by=['rs_name','test_fold',"val_fold"]).reset_index(drop=True)
                
            df_test_mean = df_all.groupby(['rs_name',"test_fold"]).mean()
            df_test_stddev = df_all.groupby(['rs_name',"test_fold"]).std()
    
            series_count_test = df_all.groupby(['rs_name',"test_fold"]).count()["val_fold"]
            df_test_stderror = df_test_stddev.div(series_count_test**0.5, axis=0)

            
            p_path_folder_output = Path(config["output_path"])

            p_path_folder_output.mkdir(parents=True, exist_ok=True)
            
            prefix_output = config["prefix_filename"]
            
            name_file = f"{prefix_output}_all.csv"           
            path_fileoutput = p_path_folder_output / name_file
            
            df_all.to_csv(path_fileoutput)
            print(colored(f"Complete results saved in: {path_fileoutput}.", 'green'))

            name_file = f"{prefix_output}_mean_per_test_fold.csv"
            path_fileoutput = p_path_folder_output / name_file

            df_test_mean.to_csv(path_fileoutput)
            print(colored(f"Mean results saved in: {path_fileoutput}.", 'green'))

            name_file = f"{prefix_output}_stderr_per_test_fold.csv"
            path_fileoutput = p_path_folder_output / name_file

            df_test_stderror.to_csv(path_fileoutput)
            print(colored(f"Standard error results saved in: {path_fileoutput}.", 'green'))
            
            
    # if config["is_outer"] == True:
    else:
        l_test_folds = dict_prediction_index.keys()
            
        # calculate metrics
        dict_metrics = {'accuracy': accuracy_score, 'balanced_accuracy': balanced_accuracy_score,
                        'f1': f1_score}

        df_all = pd.DataFrame()
    
        for test_fold in l_test_folds:
            dict_row = {"test_fold":[test_fold]}
            for metric in dict_metrics:
                true_vals = dict_true_index[test_fold]
                pred_vals = dict_prediction_index[test_fold]
                
                if metric == 'f1':
                    metric_value = dict_metrics[metric](y_true=true_vals,
                                        y_pred=pred_vals, average="weighted")
                else:
                    metric_value = dict_metrics[metric](y_true=true_vals,
                                        y_pred=pred_vals)
                
                dict_row[metric] = metric_value
                
            df_temp = pd.DataFrame(dict_row)
            # join them in pandas dataframe 
            df_all = pd.concat([df_all, df_temp], ignore_index=True)
                
        df_all=df_all.sort_values(by=['test_fold']).reset_index(drop=True)
        
        n_testfolds = df_all.shape[0]
        df_all = df_all.set_index('test_fold')
    
        df_all.loc['mean'] = df_all.mean()
        df_all.loc['std_err'] = df_all.std()/n_testfolds**0.5
        
        p_path_folder_output = Path(config["output_path"])

        p_path_folder_output.mkdir(parents=True, exist_ok=True)
        
        prefix_output = config["prefix_filename"]
        
        name_file = f"{prefix_output}_all.csv"           
        path_fileoutput = p_path_folder_output / name_file
        
        df_all.to_csv(path_fileoutput)
        print(colored(f"Complete results for {metric} saved in: {path_fileoutput}.", 'green'))

    
def main(config=None):
    """ The main program.

    Args:
        config (dict): A JSON configuration as a dictionary. (Optional)
    """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('./results_processing/summary_table/summary_table_config.json')

    if config["is_outer"] == True and config["random_search_if_inner"] == True:
        raise ValueError("config['is_outer'] cannot be true when config['random_search_if_inner'] is true")

    if config["is_outer"] == False:
        
        l_true_label_index, l_prediction_index = get_list_true_and_prediction(config)

        dict_true_label_index = read_data_inner(l_true_label_index, config)
        dict_prediction_index = read_data_inner(l_prediction_index, config)

        len_true = len(dict_true_label_index)
        len_prediction = len(dict_prediction_index)

        if len_true != len_prediction:
            raise RuntimeError(f"Number of keys is dict_true_label_index: {len_true} and" + 
                   f" Number of keys is dict_prediction_index: {len_prediction}."
                   +" They are different.")

        save_dataframes(dict_true_label_index, dict_prediction_index, config)
    
    # config["is_outer"] == True
    else:

        l_true_label_index, l_prediction_index = get_list_true_and_prediction_outer(config)

        dict_true_label_index = read_data_outer(l_true_label_index)
        dict_prediction_index= read_data_outer(l_prediction_index)

        save_dataframes(dict_true_label_index, dict_prediction_index, config) 
        

if __name__ == "__main__":
    """ Executes the program """
    main()
