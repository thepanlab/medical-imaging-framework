from termcolor import colored
import pandas as pd
import path_getter
import argparse
import json
import os 

"""
    Predicted Formatter

    This file will take the model predictions, as a matrix of probabilities,
    and translate them to a vector of integer values.
    It will do this for every prediction-file, so it expects a specific structure and naming scheme.

    > data_path/
        > Test_subject_e1/                              # Level 1
            > config_1/                                 # Level 2
                > [some fold name]                      # Level 3
                    > prediction
                        > [some fold name]_val_predicted.csv        <-- Target input
                        > [some fold name]_val_predicted_index.csv  <-- Target output
                        > ...
                > ...
            > config_2/
            > config_3/
            > winner_model/
        > Test_subject_e2/
        ...

"""



def read_json():
    """ Reads in the JSON config file """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-j', '--json', '--load_json',
        help='Load settings from a JSON file.',
        required=False,
        default='predicted_formatter_config.json'
    )
    args = parser.parse_args()
    with open(args.json) as config_file:
        return json.load(config_file)



def translate_file(file_path):
    """ Reads and translates a data-file """

    # Get the data in the CSV file
    data = pd.read_csv(file_path, header=None)
    
    # Loop through all the rows, note the largest probability-value's index
    formatted_data = []
    for i, values in data.iterrows():
        values = [v for v in values]
        index = values.index(max(values))
        formatted_data.append(index)

    # Return a list of max-indexes
    return formatted_data



def write_file(file_path, formatted_data):
    """ Writes the formatted data out to CSV """

    # Append "_index" to the input file name
    new_file_path = file_path[:len(file_path) - 4] + '_index.csv'

    # Open file and write the data to it
    with open(new_file_path, 'w') as file:
        for item in formatted_data:
            file.write(f"{item}\n")



def main():
    """ The Main Program """
    # Read the JSON file's arguements
    config = read_json()
    data_path = config["data_path"]

    # Get a list of all files to translate
    file_paths = path_getter.get_files(data_path, "prediction", isIndex=False)

    # Translate each file
    for fold in file_paths:
        for file_path in file_paths[fold]:

            # The translation
            translated_data = translate_file(file_path)

            # Write it to file
            write_file(file_path, translated_data)

    print(colored("Formatting is finished!\n", "green"))

    

if __name__ == "__main__":
    """ Executes the code """
    main()
