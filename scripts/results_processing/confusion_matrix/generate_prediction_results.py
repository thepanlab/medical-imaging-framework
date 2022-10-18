from results_processing.confusion_matrix import confusion_matrix
from util.get_config import parse_json
from termcolor import colored
from util import path_getter
import pandas as pd
import traceback
import os


def find_directories(data_path):
    """ Finds the directories for every input needed to make graphs.

    Args:
        data_path (string): The path of the data directory.

    Returns:
        dict: Two dictionaries of prediction and truth paths.
    """
    # Get the paths of every prediction and true CSV, as well as the fold-names
    is_outer = path_getter.is_outer_loop(data_path)
    pred_paths = path_getter.get_subfolder_files(data_path, "prediction", isIndex=True, getValidation=True)
    true_paths = path_getter.get_subfolder_files(data_path, "true_label", isIndex=True, getValidation=True)
    return pred_paths, true_paths, is_outer


def print_results(pred_paths, true_paths, output_path, labels, is_outer):
    """ Run the program for each item.

    Args:
        args (dict): A JSON configuration input as a dictionary.
        pred_paths (dict): A dictionary of prediction paths for the given data directory.
        true_paths (dict): A dictionary of truth paths for the given data directory.
    """    
    if is_outer:
        output_path = os.path.abspath(os.path.join(output_path, 'outer_loop'))
    else:
        output_path = os.path.abspath(os.path.join(output_path, 'inner_loop'))
        
    cols = [
        'pred_path', 'pred_label', 'pred_index', 
        'true_path', 'true_label', 'true_index', 
        'correct', 'test_subject', 'val_fold'
    ]

    # For each item, run the program
    for model in pred_paths:
        for subject in pred_paths[model]:
            df = pd.DataFrame(columns=cols)
            for i in range(len(pred_paths[model][subject])):
                pred_path = pred_paths[model][subject][i]
                true_path = true_paths[model][subject][i]
                pred_vals = pd.read_csv(pred_path, header=None).to_numpy()
                true_vals = pd.read_csv(true_path, header=None).to_numpy()
                
                
def get_results(pred_vals, true_vals, labels):
    correctness = []
    pred_labels = []
    true_labels = []
    for val_index in range(len(pred_vals)):
        pred_val = pred_vals[val_index]
        true_val = true_vals[val_index]
        correctness.append(pred_val == true_val)
        pred_labels.append(labels[pred_val])
        true_labels.append(labels[true_val])
    return correctness, pred_labels, true_labels
        
                
                


def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json(os.path.abspath('confusion_matrix_many_config.json'))
    pred_paths, true_paths, is_outer = find_directories(config["data_path"])
    print_results(pred_paths, true_paths, is_outer)


if __name__ == "__main__":
    """ Executes Program. """
    main()
