from util.get_config import parse_json
from util import path_getter
from . import roc_curve


def run_program(args):
    """ Run the program for each item.

    Args:
        args (dict): Program configuration.
    """
    # Get the needed input paths, as well as the proper file names for output
    pred_paths, true_paths = find_directories(args["data_path"], args['is_outer'])
    json = {
        label: args[label] for label in (
            'line_width', 'label_types', 'line_colors','alpha',
            'font_family', 'label_font_size', 'title_font_size',
            'save_resolution', 'save_format', 'output_path'
        )
    }

    # For each item, run the program
    for model in pred_paths:
        for subject in pred_paths[model]:
            for item in range(len(pred_paths[model][subject])):
                # Get the program's arguments
                json = generate_json(pred_paths, true_paths, model, subject, item, json)
                roc_curve.main(json)


def find_directories(data_path, is_outer):
    """ Finds the directories for every input needed to make graphs.

    Args:
        data_path (str): Path to the data.

    Returns:
        dict: Of the true paths and the prediction paths.
    """
    # Get the paths of every prediction and true CSV, as well as the fold-names
    true_paths = path_getter.get_subfolder_files(data_path, "true_label", isIndex=True, getValidation=True, isOuter=is_outer)
    pred_paths = path_getter.get_subfolder_files(data_path, "prediction", isIndex=False, getValidation=True, isOuter=is_outer)
    return pred_paths, true_paths


def generate_json(pred_paths, true_paths, model, subject, item, json):
    """ Creates a dictionary of would-be JSON arguments

    Args:
        pred_paths (dict): Paths to the predicted values.
        true_paths (dict): Paths to the true values.
        model (str): The model name used.
        subject (str): The subject name.
        item (int): The array index.
        json (dict): The 'json' to add to.

    Returns:
        dict: A 'json' configuration.
    """
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
    config = parse_json('./results_processing/roc_curve/roc_curve_many_config.json')
    run_program(config)


if __name__ == "__main__":
    """ Executes Program. """
    main()