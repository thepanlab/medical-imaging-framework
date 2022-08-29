from get_config import parse_json
from termcolor import colored
import ultraimport
import path_getter
import subprocess
import os

# Imports a module from a level above.
# If moved to same level, use "import confusion_matrix".
confusion_matrix = ultraimport('../confusion_matrix.py')



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
    for key in pred_paths:
        for index in range(len(pred_paths[key])):
            try:
                # Get the program's arguments
                json = generate_json(pred_paths, true_paths, key, index, json)
                confusion_matrix.main(json)
            # Catch weird stuff
            except Exception as err:
                print(colored("Exception caught.\n\t" + str(err) + "\n", "red"))



def find_directories(data_path):
    """ Finds the directories for every input needed to make graphs. """
    # Get the paths of every prediction and true CSV, as well as the fold-names
    pred_paths = path_getter.get_files(data_path, "prediction", isIndex=True)
    true_paths = path_getter.get_files(data_path, "true_label", isIndex=True)
    return pred_paths, true_paths



def generate_json(pred_paths, true_paths, key, index, json):
    """ Creates a dictionary of would-be JSON arguments """
    # The current expected suffix format for true labels
    true_label_suffix = " true label index.csv"

    # Create dictionary for every item
    json["pred_path"] = pred_paths[key][index]
    json["true_path"] = true_paths[key][index]
    json["output_file_prefix"] = true_paths[key][index].split('/')[-1].replace(true_label_suffix, "")
    return json



def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json('confusion_matrix_graphing_config.json')
    run_program(config)



if __name__ == "__main__":
    """ Executes Program. """
    main()