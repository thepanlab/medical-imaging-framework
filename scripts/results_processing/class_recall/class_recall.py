from sklearn.metrics import accuracy_score
from util.get_config import parse_json
from scipy.stats import sem as std_err
from termcolor import colored
from util import path_getter
import pandas as pd
import numpy as np
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
                try:
                    results[config][test_fold][val_fold] = pd.read_csv(validation_fold_path, header=None).to_numpy()
                except:
                    print(colored(f"Warning: {validation_fold_path} is empty.", 'yellow'))

    # Return the dictionary
    return results


def get_recall_and_stderr(true, pred, classes):
    """ Gets the recall of each fold-config index.

    Args:
        true (dict): A dictionary of true values sorted by config and test fold.
        pred (dict): A dictionary of predicted values sorted by config and test fold.

    Returns:
        dict: Two dictionaries of relative and real recall and standard error. Sorted by config and test fold.
    """
    # Get a table of accuracy for every config
    recall = {'individual': {}, 'column': {}}
    for config in pred:
        recall['individual'][config] = {}
        recall['column'][config] = {}
        
        # Get the accuracy  of every test-val fold pair
        for test_fold in pred[config]:
            recall['individual'][config][test_fold] = {}
            for val_fold in pred[config][test_fold]:
                recall['individual'][config][test_fold][val_fold] = {}
                
                # Store the predicted and true values
                pred_vals = pred[config][test_fold][val_fold]
                true_vals = true[config][test_fold][val_fold]
                
                # Loop through the classes
                for class_label in classes:
                    recall['individual'][config][test_fold][val_fold][class_label] = 0
                    if class_label not in recall['column'][config]:
                        recall['column'][config][class_label] = []
                    
                    # Find where this class is in the new labels
                    label_indexes = list(np.where(true_vals == classes[class_label])[0])
                    
                    # Make arrays of true and predicted values for this class label
                    pred_class_vals = [int(pred_vals[i]) for i in label_indexes]
                    true_class_vals = [int(true_vals[i]) for i in label_indexes]
                    
                    # Get the accuracy of these class-values
                    recall['individual'][config][test_fold][val_fold][class_label] = accuracy_score(y_true=true_class_vals, y_pred=pred_class_vals)                
                                                          
                    # Total class-label accuracy mean-sum
                    recall['column'][config][class_label] += [recall['individual'][config][test_fold][val_fold][class_label]]
                
        # Get the average column recall
        for class_label in recall['column'][config]:
            denom = len(recall['column'][config][class_label])

            recall['column'][config][class_label] = sum(recall['column'][config][class_label]) / denom
            
    # Calculate the standard error of the recall
    errs = {}
    for config in pred:
        errs[config] = {}
        
        # Get the standard error of the test-fold recall
        for test_fold in pred[config]:
            errs[config][test_fold] = {}
            for val_fold in pred[config][test_fold]:
                errs[config][test_fold][val_fold] = {}
                for class_label in recall['individual'][config][test_fold][val_fold]:
                    errs[config][test_fold][val_fold][class_label]  = std_err(recall['individual'][config][test_fold][val_fold][class_label])
                
    # Return accuracy
    return recall, errs


def total_output(recall, standard_error, classes, output_path, output_file, round_to, is_outer):
    """ Produces a table of recall and standard errors by config and fold

    Args:
        recall (dict): A dictionary of recall sorted by config and test fold.
        standard_error (dict): A dictionary of errors sorted by config and test fold.
        output_path (str): A string of the directory the output CSV should be written to.
        output_file (str): Name prefix of the output files.
        round_to (int): Gives the max numerical digits to round a value to.
        is_outer (bool): If the data is from the outer loop.
    """
    # Get names of columns and subjects
    configs = list(recall['individual'].keys())
    test_folds = list(recall['individual'][configs[0]].keys())
    val_folds = list(recall['individual'][configs[0]][test_folds[0]].keys())
    test_folds.sort()
    val_folds.sort()
    
    # Get output path
    if is_outer:
        output_path = os.path.join(output_path, 'outer_loop')
    else:
        output_path = os.path.join(output_path, 'inner_loop')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Create a table for each config
    for config in configs:
        df = pd.DataFrame(columns=['Test Fold', 'Validation Fold'] + [c for c in classes])
        
        # For every test and val combo, get class recall
        for test_fold in test_folds:       
            for val_fold in recall['individual'][config][test_fold]:
                row = {
                    'Test Fold': test_fold,
                    'Validation Fold': val_fold
                }   
                
                # Add accuracy for every class
                for class_label in classes:
                    if class_label in recall['individual'][config][test_fold][val_fold]:
                        row[class_label] = round(recall['individual'][config][test_fold][val_fold][class_label], round_to)
                    else:
                        row[class_label] = -1
            
                # Append the row to the dataframe
                df = df.append(pd.DataFrame(row, index=[0]).loc[0])
            
        # Add a row of the mean recall
        row = {
            'Test Fold': 'Average',
            'Validation Fold': None
        } 
        for class_label in classes:
            if class_label in recall['column'][config]:
                row[class_label] = round(recall['column'][config][class_label], round_to)
            else:
                row[class_label] = -1
        df = df.append(pd.DataFrame(row, index=[0]).loc[0])
            
        # Create and save the Pandas dataframe
        if is_outer:
            df.to_csv(f'{output_path}/{output_file}_{config}_outer.csv')
        else:
            df.to_csv(f'{output_path}/{output_file}_{config}_inner.csv')


def main(config=None):
    """ The main program.

    Args:
        config (dict): A JSON configuration as a dictionary. (Optional)
    """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('./results_processing/class_accuracy/class_accuracy_config.json')

    # Get the necessary input files
    true_paths = path_getter.get_subfolder_files(config['data_path'], "true_label", isIndex=True, getValidation=True, isOuter=config['is_outer'])
    pred_paths = path_getter.get_subfolder_files(config['data_path'], "prediction", isIndex=True, getValidation=True, isOuter=config['is_outer'])

    # Read in each file into a dictionary
    true = read_data(true_paths)
    pred = read_data(pred_paths)
    
    # Get recall and standard error
    recalls, stderr = get_recall_and_stderr(true, pred, config['classes'])

    # Output results
    total_output(recalls, stderr, config['classes'], config['output_path'], config['output_filename'], config['round_to'], is_outer=False)
    print(colored("Finished writing the class recall.", 'green'))


if __name__ == "__main__":
    """ Executes the program """
    main()
