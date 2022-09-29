import argparse
import json


def parse_json(default_config_file_name):
    """ Reads in the configuration from a JSON file. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-j', '--json', '--load_json',
        help='Load settings from a JSON file.',
        required=False,
        default=default_config_file_name
    )
    args = parser.parse_args()
    with open(args.json) as config_file:
        config = json.load(config_file)
    return config
