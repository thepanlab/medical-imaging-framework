from cgi import test
from get_config import parse_json
from termcolor import colored
from statistics import mode
import confusion_matrix
import pandas as pd
import numpy as np
import path_getter
import regex as re
import math
import os


def run_program(args, pred_paths, true_paths):
    """ Run the program for each item. """
    # Get the proper file names for output
    json = {
        label: args[label] for label in (
            'label_types', 'output_path'
        )
    }

    # For each item, run the program
    for model in pred_paths:
        for subject in pred_paths[model]:
            for item in range(len(pred_paths[model][subject])):
                try:
                    # Get the program's arguments
                    json = generate_json(pred_paths, true_paths, model, subject, item, json)
                    confusion_matrix.main(json)

                # Catch weird stuff
                except Exception as err:
                    print(colored("Exception caught.\n\t" + str(err) + "\n", "red"))
    return pred_paths


def find_directories(data_path):
    """ Finds the directories for every input needed to make graphs. """
    # Get the paths of every prediction and true CSV, as well as the fold-names
    pred_paths = path_getter.get_subfolder_files(data_path, "prediction", isIndex=True, getValidation=True)
    true_paths = path_getter.get_subfolder_files(data_path, "true_label", isIndex=True, getValidation=True)
    return pred_paths, true_paths


def generate_json(pred_paths, true_paths, model, subject, item, json):
    """ Creates a dictionary of would-be JSON arguments """
    # The current expected suffix format for true labels
    true_label_suffix = " true label index.csv"

    # Create dictionary for every item
    json["pred_path"] = pred_paths[model][subject][item]
    json["true_path"] = true_paths[model][subject][item]
    json["output_file_prefix"] = true_paths[model][subject][item].split('/')[-1].replace(true_label_suffix, "")
    return json


def get_mean_matrices(pred_paths, input_path, output_path, labels):
    """ This function gets the mean confusion matrix of every inner loop """
    # Get the confusion matrix paths that were created earlier
    matrix_paths = [os.path.join(input_path, m).replace("\\", "/") for m in os.listdir(input_path)]

    # Check shapes
    shapes = {}
    shapes_list = []
    folds = []
    for config in pred_paths:
        shapes[config] = {}
        for test_fold in pred_paths[config]:
            if test_fold not in folds:
                folds.append(test_fold)
            shapes[config][test_fold] = {}
            for val_fold in pred_paths[config][test_fold]:
                if val_fold not in folds:
                    folds.append(val_fold)

                # Read CSV to get shape
                pred_rows, pred_cols = pd.read_csv(val_fold, header=None).to_numpy().shape
                val_id = val_fold.split("/")[-3].split("_")[-1]
                shapes[config][test_fold][val_id] = pred_rows
                shapes_list.append(pred_rows)

    # Get mode row-length of ALL matricies
    shapes_mode = mode(shapes_list)

    # Get the mean of each test fold + configuration
    for config in pred_paths:
        for test_fold in pred_paths[config]:

            # The matrices to combine, within this config and testing fold
            subset = [path for path in matrix_paths if config in path.split('/')[-1]]
            subset = [path for path in subset if ("test_" + test_fold + "_val") in path.split('/')[-1]]

            # Check shape is the mode for each validation fold
            for val_id in shapes[config][test_fold]:
                if not shapes_mode == shapes[config][test_fold][val_id]:
                    print(colored(f"Warning: Not taking the avg/stderr of the matrix of test fold {test_fold} and validation fold {val_id} in {config}" +
                        f"\n\tThe expected shape was ({shapes_mode}, {shapes_mode}), but got ({shapes[config][test_fold][val_id]}, {shapes[config][test_fold][val_id]})." , "yellow"))
                    file = [path for path in subset if config in path.split('/')[-1]]
                    file = [path for path in subset if (f"test_{test_fold}_val_{val_id}") in path.split('/')[-1]][0]
                    subset.remove(file)

            # Check if length is valid for finding mean/stderr
            if len(subset) <= 1:
                print(colored(f"Warning: Mean/stderr calculation skipped for testing fold {test_fold} in {config}."
                              + " Must have multiple validation folds.", 'yellow'))
                continue

            # Read in all the relevant matrices
            mats = []
            for path in subset:
                mat = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
                mats.append(mat)

            # Compute the average matrix
            avg_matrix_df = pd.DataFrame(0, columns=labels, index=labels)
            avg_matrix_df.index = [["Truth"] * len(labels), avg_matrix_df.index]
            avg_matrix_df.columns = [["Predicted"] * len(labels), avg_matrix_df.columns]
            for col in labels:
                for row in labels:
                    for mat in mats:
                        avg_matrix_df["Predicted"][col]["Truth"][row] += mat["Predicted"][col]["Truth"][row]
                    avg_matrix_df["Predicted"][col]["Truth"][row] /= len(mats)

            # Compute the standard error of the matrices
            stderr_matrix_df = pd.DataFrame(0, columns=labels, index=labels)
            stderr_matrix_df.index = [["Truth"] * len(labels), stderr_matrix_df.index]
            stderr_matrix_df.columns = [["Predicted"] * len(labels), stderr_matrix_df.columns]
            for col in labels:
                for row in labels:
                    for mat in mats:
                        stderr_matrix_df["Predicted"][col]["Truth"][row] += \
                            (mat["Predicted"][col]["Truth"][row] - avg_matrix_df["Predicted"][col]["Truth"][row]) ** 2
                    stderr_matrix_df["Predicted"][col]["Truth"][row] = \
                        math.sqrt(mat["Predicted"][col]["Truth"][row] / (len(mats) - 1))

            # Output both matrices
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            avg_matrix_df.to_csv(os.path.join(output_path, f'{config}_{test_fold}_avg_conf_matrix.csv'))
            stderr_matrix_df.to_csv(os.path.join(output_path, f'{config}_{test_fold}_stderr_conf_matrix.csv'))
            print(colored(f"Average and stderr confusion matrix created for {test_fold} in {config} \n", 'green'))

    return shapes


"""
def get_attribute_overview_matrices(data, input_path, output_path, labelA, labelB):
     This function gets the mean confusion matrix from two specific labels 
    # Get the confusion matrix paths that were created earlier
    matrix_paths = [os.path.join(input_path, m).replace("\\", "/") for m in os.listdir(input_path)]

    # To fill in
    results = {}
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get all possible folds (in case uneven)
    folds = list(data[list(data.keys())[0]].keys())
    folds.extend(list(data[list(data.keys())[0]][folds[0]]))
    folds = sorted(set(folds))

    # Put values into dataframe
    rownames = []
    colnames = []
    for fold in folds:
        rownames.append(fold)
        colnames.extend([fold +'_avg', fold +'_err'])

    for config in data:

        results[config] = pd.DataFrame(0.0, columns=colnames, index=rownames)

        # Get the value of every val test fold 
        for test_fold in data[config]:
            
            # Read in the specific matrix
            in_mat = [mat for mat in matrix_paths if config in mat.split('/')[-1]]
            in_mat = [mat for mat in in_mat if (f"_{test_fold}_") in mat.split('/')[-1]]
            if not in_mat:
                continue
            mat_stderr = pd.read_csv([mat for mat in in_mat if (f"stderr") in mat.split('/')[-1]][0], header=[0, 1], index_col=[0, 1])
            mat_avg = pd.read_csv([mat for mat in in_mat if (f"avg") in mat.split('/')[-1]][0], header=[0, 1], index_col=[0, 1])

            for val_fold in data[config][test_fold]
                # Extract the target values from the matricies
                vals = {
                    "err": mat_stderr["Predicted"][labelA]["Truth"][labelB],
                    "avg": mat_avg["Predicted"][labelA]["Truth"][labelB]
                }
                results[config][f"{test_fold}_avg"][test_fold] = vals['avg']
                results[config][f"{test_fold}_err"][test_fold] = vals['err']

            print(results[config])

        # Create the extra-index/col names
        results[config].index = [["Truth"] * len(rownames), results[config].index]
        results[config].columns = [["Predicted"] * len(colnames), results[config].columns]

        # Output both matrices
        results[config].to_csv(os.path.join(output_path, f'{config}_{labelA}_{labelB}_matrix.csv'))
        print(colored(f"Average and stderr value matrix for specified labels were created for {labelA} and {labelB} in {config} \n", 'green'))
"""


def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json('confusion_matrix_graphing_config.json')
    pred_paths, true_paths = find_directories(config["data_path"])
    run_program(config, pred_paths, true_paths)
    data = get_mean_matrices(pred_paths, config['output_path'], config['average_stderr_output_path'], config['label_types'])
    
    # Compare label results
    #for pair in config['label_comparisons']:
    #    get_attribute_overview_matrices(data, config['output_path'], config['label_stderr_output_path'], pair[0], pair[1])

if __name__ == "__main__":
    """ Executes Program. """
    main()
