from get_config import parse_json
from termcolor import colored
import path_getter
import roc_curve


def run_program(args):
    """ Run the program for each item. """
    # Get the needed input paths, as well as the proper file names for output
    pred_paths, true_paths = find_directories(args["data_path"])
    json = {
        label: args[label] for label in (
            'line_width', 'label_types', 'line_colors',
            'font_family', 'label_font_size', 'title_font_size',
            'save_resolution', 'save_format', 'output_path'
        )
    }

    # For each item, run the program
    for model in pred_paths:
        for subject in pred_paths[model]:
            for item in range(len(pred_paths[model][subject])):
                try:
                    # Get the program's arguments
                    json = generate_json(pred_paths, true_paths, model, subject, item, json)
                    roc_curve.main(json)

                # Catch weird stuff
                except Exception as err:
                    raise err
                    #print(colored("Exception caught.\n\t" + str(err) + "\n", "red"))


def find_directories(data_path):
    """ Finds the directories for every input needed to make graphs. """
    # Get the paths of every prediction and true CSV, as well as the fold-names
    true_paths = path_getter.get_subfolder_files(data_path, "true_label", isIndex=True)
    pred_paths = path_getter.get_subfolder_files(data_path, "prediction", isIndex=False)
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


def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json('roc_curve_graphing_config.json')
    run_program(config)


if __name__ == "__main__":
    """ Executes Program. """
    main()
