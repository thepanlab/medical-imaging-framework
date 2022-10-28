from results_processing.confusion_matrix import confusion_matrix
from util.get_config import parse_json
from termcolor import colored
from util import path_getter
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


def run_program(args, pred_paths, true_paths, is_outer):
    """ Run the program for each item.

    Args:
        args (dict): A JSON configuration input as a dictionary.
        pred_paths (dict): A dictionary of prediction paths for the given data directory.
        true_paths (dict): A dictionary of truth paths for the given data directory.
    """
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
                    print(colored(f"Exception caught.\n\t{str(err)}", "red"))
                    print(colored(f"{traceback.format_exc()}\n", "yellow"))


def generate_json(pred_paths, true_paths, config, subject, item, json):
    """ Creates a dictionary of would-be JSON arguments

    Args:
        pred_paths (dict): A dictionary of prediction paths.
        true_paths (dict): A dictionary of truth paths.
        config (str): The config (model) of the input data.
        subject (str): The subject (test fold) of the input data.
        item (int): The item's index in the paths dictionary-array.
        json (dict): A dictionary with some values already added.

    Returns:
        dict: A JSON configuration as a dict.
    """
    # The current expected suffix format for true labels
    true_label_suffix = " true label index.csv"

    # Create dictionary for every item
    json["pred_path"] = pred_paths[config][subject][item]
    json["true_path"] = true_paths[config][subject][item]
    json["output_file_prefix"] = true_paths[config][subject][item].split('/')[-1].replace(true_label_suffix, "")
    return json


def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json(os.path.abspath('./results_processing/confusion_matrix/confusion_matrix_many_config.json'))
    pred_paths, true_paths, is_outer = find_directories(config["data_path"])
    run_program(config, pred_paths, true_paths, is_outer)


if __name__ == "__main__":
    """ Executes Program. """
    main()
