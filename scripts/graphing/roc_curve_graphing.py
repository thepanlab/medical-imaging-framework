from termcolor import colored
import ultraimport
import path_getter
import subprocess
import argparse
import json
import os

# Imports a module from a level above.
# If moved to same level, use "import roc_curve".
roc_curve = ultraimport('../roc_curve.py')



def get_config():
    """ Reads in the configuration from a JSON file. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-j', '--json', '--load_json',
        help='Load settings from a JSON file.',
        required=False,
        default='roc_curve_graphing_config.json'
    )
    args = parser.parse_args()
    with open(args.json) as config_file:
        config = json.load(config_file)
    return config



def run_program(args):
    """ Run the program for each item. """

    # Get the needed input paths, as well as the proper file names for output
    pred_paths, true_paths = find_directories(args["data_path"])
    json_dicts = generate_jsons(pred_paths, true_paths, args)

    # For each item, run the program
    for json in json_dicts:
        try:
            # Get the program's arguments
            roc_curve.main(json)

        # Catch weird stuff
        except Exception as err:
            print(colored("Exception caught.\n\t" + str(err) + "\n", "red"))



def find_directories(data_path):
    """ Finds the directories for every input needed to make graphs. """
    # Get the paths of every prediction and true CSV, as well as the fold-names
    pred_paths = path_getter.get_files(data_path, "prediction", isIndex=False)
    true_paths = path_getter.get_files(data_path, "true_label", isIndex=True)
    
    # Return what was found
    return pred_paths, true_paths



def generate_jsons(pred_paths, true_paths, args):
    """ Creates a dictionary of would-be JSON arguments """
    # Will return this
    dicts = []

    # Create dictionary for every item
    for key in pred_paths:
        for i in range(len(pred_paths[key])):
            json_args = {}
            json_args["pred_path"]   = pred_paths[key][i]
            json_args["true_path"]   = true_paths[key][i]
            json_args["output_path"] = args["output_path"]
            json_args["output_file_prefix"] = true_paths[key][i].split('/')[-1].replace(" true label index.csv", "")
            
            json_args["line_width"]  = args["line_width"]
            json_args["label_types"] = args["label_types"]
            json_args["line_colors"] = args["line_colors"]

            json_args["font_family"] = args["font_family"]
            json_args["label_font_size"] = args["label_font_size"]
            json_args["title_font_size"] = args["title_font_size"]

            json_args["save_resolution"] = args["save_resolution"]
            json_args["save_format"] = args["save_format"]
            
            dicts.append(json_args)

    # Return the copied dictionary
    return dicts



def main():
    """ The Main Program. """
    # Get program configuration
    config = get_config()

    # Write JSON files for each image to process
    run_program(config)



if __name__ == "__main__":
    """ Executes Program. """
    main()
