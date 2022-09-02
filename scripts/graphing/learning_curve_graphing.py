from get_config import parse_json
from termcolor import colored
import ultraimport
import path_getter
import subprocess
import os

# Imports a module from a level above.
# If moved to same level, use "import learning_curve".
learning_curve = ultraimport('../learning_curve.py')



def run_program(args):
    """ Run the program for each item. """
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
    for subfold in subfold_paths:
        try:
            # Get the program's arguments
            json['input_path'] = subfold
            learning_curve.main(json)
        # Catch weird stuff
        except Exception as err:
            print(colored("Exception caught.\n\t" + str(err) + "\n", "red"))


def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json('learning_curve_graphing_config.json')
    run_program(config)



if __name__ == "__main__":
    """ Executes Program. """
    main()
