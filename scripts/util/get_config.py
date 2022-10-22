from contextlib import redirect_stderr
from termcolor import colored
import argparse
import json
import io


def parse_json(default_config_file_name):
    """ Parses the config file from the command line. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-j', '--json', '--load_json',
        help='Load settings from a JSON file.',
        required=False
    )
    try:
        shh = io.StringIO()
        with redirect_stderr(shh):
            args = parser.parse_args()
    except:
        return prompt_json(default_config_file_name)
    with open(args.json) as config:
        return json.load(config)


def prompt_json(default_config_file_name):
    """ Prompts the user for a config name. """
    config_file = input(colored(
        "Please enter the config path, or press enter to use the default path:\n",
        'cyan'
    ))
    if not config_file:
        config_file = default_config_file_name
    with open(config_file) as config:
        return json.load(config)
