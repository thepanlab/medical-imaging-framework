from termcolor import colored
import pandas as pd
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
                > [some model name]_0_test_e1_val_e2    # Level 3
                    > prediction
                        > [some model name]_0_test_e1_val_e2_test_predicted.csv   <-- Target files
                        > [some model name]_0_test_e1_val_e2_val_predicted.csv    <--
                    > ...
                > ...
            > config_2/
            > config_3/
            > winner_model/
        > Test_subject_e2/
        ...

"""



""" Reads in the JSON config file """
def read_json():
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



""" Get paths of files to translate """
def get_file_paths(data_path):

    # Add to this
    file_paths = []
    
    # Check if the directory exists
    if not os.path.exists(data_path):
        raise Exception("Error: The data path does not exist! - " + data_path)
    
    # Get the subdirectories, they *should* be for each available test subject 
    subdirs = os.listdir(data_path)
    subdirs = [os.path.join(data_path, s) for s in subdirs]

    # In each subdirectory (Level 1), get the predicted files and translate them
    for dir in subdirs:


        # Double-check this is a test subject directory
        if "Test_subject" not in dir:
            print(colored(f"Warning: non-test subject directory detected in the data path - \n\t{dir} \n\n", 'yellow'))
            continue

        # Get the subdirectories (Level 2) of this directory
        subsubdirs = os.listdir(dir)
        subsubdirs = [os.path.join(dir, s) for s in subsubdirs]
        for subdir in subsubdirs:

            # Get the subdirectories (Level 3) of this sub-subdirectory
            subsubsubdirs = os.listdir(subdir)
            subsubsubdirs = [os.path.join(subdir, s) for s in subsubsubdirs]
            if len(subsubsubdirs) == 0:
                print(colored(f"Warning: empty directory detected - \n\t{subdir} \n\n", 'yellow'))
                continue

            # Search each of these directories for a prediction folder
            for subsubdir in subsubsubdirs:

                # Get folders inside
                folders = os.listdir(subsubdir)
                folders = [os.path.join(subsubdir, s) for s in folders]
                if len(folders) == 0:
                    print(colored(f"Warning: empty directory detected - \n\t{subsubdir} \n\n", 'yellow'))
                    continue

                # Check prediction folder exists
                target = os.path.join(subsubdir, "prediction")
                if target not in folders:
                    print(colored(f"Warning: directory does not contain a 'prediction' folder - \n\t{subsubdir} \n\n", 'yellow'))
                    continue

                # Get files in the prediction folder
                files = os.listdir(target)
                files = [os.path.join(target, s) for s in files]
                for f in files:

                    # Check if not a prediction file
                    isPredicted = f.endswith("predicted.csv")
                    isFormatted = f.endswith("predicted_index.csv") 
                    if not (isPredicted or isFormatted):
                        print(colored(f"Warning: prediction-files should end in '_predicted.csv' - \n\t{f} \n\n", 'yellow'))
                        continue

                    # Place only not-formatted files in the path list
                    if isPredicted:
                        file_paths.append(f)

    # Finally, return all of the valid paths
    return file_paths



""" Reads and translates a data-file """
def translate_file(file_path):

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



""" Writes the formatted data out to CSV """
def write_file(file_path, formatted_data):

    # Append "_index" to the input file name
    new_file_path = file_path[:len(file_path) - 4] + '_index.csv'

    # Open file and write the data to it
    with open(new_file_path, 'w') as file:
        for item in formatted_data:
            file.write(f"{item}\n")



""" The Main Program """
def main():

    try:
        # Read the JSON file's arguements
        config = read_json()
        data_path = config["data_path"]

        # Get a list of all files to translate
        file_paths = get_file_paths(data_path)

    # Catch weird stuff
    except Exception as err:
            print(colored(f"Error: Unexpected error caught.\n", "red"))
            print(colored(f"{err} \n\n" , 'red'))
            return

    # Translate each file
    for file_path in file_paths:
        try: 

            # The translation
            translated_data = translate_file(file_path)

            # Write it to file
            write_file(file_path, translated_data)

        # Catch weird stuff
        except Exception as err:
                print(colored(f"Error: Unexpected error caught.\n", "red"))
                print(colored(f"{err} \n\n" , 'red'))
                return

    print(colored("Formatting is finished!\n\n", "green"))

    

""" Executes the code """
if __name__ == "__main__":
    main()
