from util.get_config import parse_json
from . import learning_curve
from util import path_getter


def run_program(args):
    """ Run the program for each item.

    Args:
        args (dict): The configuration arguements.
    """
    # Get the needed input paths, as well as the proper file names for output
    subfold_paths = path_getter.get_subfolds(args["data_path"])
    json = {
        label: args[label] for label in (
            'loss_line_color', 'val_loss_line_color', 'acc_line_color', 'val_acc_line_color',
            'font_family', 'label_font_size', 'title_font_size',
            'save_resolution', 'save_format', 'output_path'
        )
    }

    # For each item, run the program
    for model in subfold_paths:
        for subject in subfold_paths[model]:
            for subfold in subfold_paths[model][subject]:
                
                # Get the program's arguments and run
                json['input_path'] = subfold
                learning_curve.main(json)


def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json('./results_processing/learning_curve/learning_curve_many_config.json')
    run_program(config)


if __name__ == "__main__":
    """ Executes Program. """
    main()
    