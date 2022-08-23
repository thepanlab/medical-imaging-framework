from termcolor import colored
import argparse
import json
import os



""" Reads in the configuration from a JSON file """
def get_config():
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



""" Writes a JSON File To The Current Folder """
def write_json(n, args):

    # for each item, write a config file
    for i in range(n):

        # copy dictionary for ith item
        json_args = {}
        json_args["pred_path"] = args["pred_paths"][i]
        json_args["true_path"] = args["true_paths"][i]
        json_args["output_path"] = args["output_paths"][i]
        json_args["output_file_prefix"] = args["output_file_prefixes"][i]
        
        json_args["line_width"] = args["line_width"]
        json_args["label_types"] = args["label_types"]
        json_args["line_colors"] = args["line_colors"]

        json_args["font_family"] = args["font_family"]
        json_args["label_font_size"] = args["label_font_size"]
        json_args["title_font_size"] = args["title_font_size"]

        json_args["save_resolution"] = args["save_resolution"]
        json_args["save_format"] = args["save_format"]

        # write to file
        json_obj = json.dumps(json_args)
        with open(args["json_file_prefix"] + str(i) + ".json", "w") as out:
            out.write(json_obj)



""" Run command-line arguments for each item """
def run_program(n, args):
    
    # for each item, run the program
    for i in range(n):
        try:
            os.system("python3 " + args["program_location"] + " -j " + args["json_file_prefix"] + str(i) + ".json")
        
        except Exception as err:
            print(colored("Exception caught, " + args["program_location"] + str(i) + ".json skipped: \n\t" + str(err) + "\n", "red"))



""" The Main Program """
def main():

    # Get program configuration
    config = get_config()
    if not os.path.exists(config['program_location']):
        raise Exception("Error: The following program does not exist!: " + config['program_location'])

    # The number of inputs to process
    n = len(config["pred_paths"])

    # Check that all input-lists are of equal length
    t = len(config["true_paths"])
    o = len(config["output_paths"])
    p = len(config["output_file_prefixes"])
    if (n != t) or (n != o) or (n != p):
        raise Exception("Error: The file-lists in the configuration file are not of equal length.")

    # Write JSON files for each image to process
    write_json(n, config)
    run_program(n, config)



""" Executes Program """
if __name__ == "__main__":
    main()
