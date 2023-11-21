from contextlib import redirect_stderr
from termcolor import colored
import argparse
import json
import io
import os


def parse_json(default_config_file_name):
    """ Parses the config file from the command line. """
    # Command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-j', '--json', '--load_json',
        help='Load settings from a JSON file.',
        required=False
    )
    
    # Try to read the arguements
    try:
        shh = io.StringIO()
        with redirect_stderr(shh):
            args = parser.parse_args()
        if args.json is None:
            raise Exception("You shouldn't see this...")
    except:
        return prompt_json(default_config_file_name)
    
    # Open the program configuration
    with open(args.json) as config:
        return json.load(config)


def prompt_json(default_config_file_name):
    """ Prompts the user for a config name. 
    Args:
        default_config_file_name (str): The default configuration location.
    """
    # Prompt for input
    config_file = input(colored(
        "Please enter the config path, or press enter to use the default path:\n",
        'cyan'
    ))
    
    # If no input, use a default file. Load in the configuration json.
    if not config_file:
        config_file = default_config_file_name
    with open(config_file) as config:
        return json.load(config)
    
    
def parse_training_configs(default_config_directory_name):
    """ Parse the configurations from the command line.
    Args:
        default_config_directory_name (str): The default configuration location.
    """
    # Command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file', '--config_file',
        type=str, default=None, required=False,
        help='Load settings from a JSON file.'
    )
    parser.add_argument(
        '--folder', '--config_folder',
        type=str, default=default_config_directory_name, required=False,
        help='Load settings from a JSON file.'
    )
    args = parser.parse_args()
    
    # Check for command line errors
    if not (args.file or args.folder):
        raise Exception("Error: no configuration file or folder specified.")
    if (args.file and args.folder) and args.folder != default_config_directory_name:
        raise Exception(colored("Error: Please only specify a configuration file or directory, not both.", 'red'))

    # Read in a single file
    if args.file:
        if not os.path.exists(args.file):
            raise Exception(colored(f"Error: The file '{args.file}' does not exist."))
        with open(args.file) as fp:
            return [json.load(fp)]
            
    # Read in a directory of files
    elif args.folder:
        if not os.path.isdir(args.folder):
            raise Exception(colored(f"Error: The directory '{args.folder}' does not exist."))
        
        # Read in each valid config file within
        configs = []
        for file in os.listdir(args.folder):
            full_path = os.path.join(args.folder, file)
            if not os.path.isdir(full_path) and full_path.endswith(".json"):
                with open(full_path) as fp:
                    configs.append(json.load(fp))
        return configs
    
    # Shouldn't reach here
    else:
        raise Exception(colored("Error: Unknown error reached in the configuration parsing.", 'red'))
    