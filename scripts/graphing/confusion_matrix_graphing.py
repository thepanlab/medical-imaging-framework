from get_config import parse_json
from termcolor import colored
import confusion_matrix
import pandas as pd
import numpy as np
import path_getter
import math
import os


def run_program(args):
    """ Run the program for each item. """
    # Get the needed input paths, as well as the proper file names for output
    pred_paths, true_paths = find_directories(args["data_path"])
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


def get_mean_matrices(path_dict, input_path, output_path, labels):
    """ This function gets the mean confusion matrix of every inner loop """
    # Get the confusion matrix paths that were created earlier
    matrix_paths = [os.path.join(input_path, m).replace("\\", "/") for m in os.listdir(input_path)]

    # Get the mean of each test fold + configuration
    for config in path_dict:
        for test_fold in path_dict[config]:

            # Subset the data by config and test fold
            subset = [path for path in matrix_paths if config in path.split('/')[-1]]
            subset = [path for path in subset if ("test_" + test_fold + "_val") in path.split('/')[-1]]

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





def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json('confusion_matrix_graphing_config.json')
    pred_paths = run_program(config)
    get_mean_matrices(pred_paths, config['output_path'], config['average_stderr_output_path'], config['label_types'])


if __name__ == "__main__":
    """ Executes Program. """
    main()
