from predicted_formatter import main as reformat
from termcolor import colored
import os

"""
    Path Getter

    This file will find all of the model predictions, true-values, and other file names.
    This will expect a particular data-path structure.

    > data_path/
        > Test_subject_e1/              # Level 1: subject
            > config_1/                 # Level 2: model
                > [some subfold name]   # Level 3: cross-validation subfold
                    > prediction        <-- This can be a target folder
                        > [some subfold name]_val_predicted.csv
                        > [some subfold name]_val_predicted_index.csv
                        > ...
                    > true_label         <-- This can be a target folder
                        > [some subfold name]_val true label.csv
                        > [some subfold name]_val true label index.csv
                        > ...
                > ...
            > config_2/
            > config_3/
            > winner_model/
        > Test_subject_e2/
        ...

"""


def get_subfiles(path, return_full_path=True):
    """ This will get a list of everything contained within some particular directory; by model and subject. """
    # Make this path have all forward-slashes, to make usable with Windows machines
    path = path.replace("\\", "/")

    # Check if the path exists and is a directory
    if not (os.path.exists(path) and os.path.isdir(path)):
        raise Exception(colored("Error: The directory does not exist! - " + path, "red"))

    # Get and return items
    subfiles = os.listdir(path)
    if return_full_path:
        subfiles = [os.path.join(path, s).replace("\\", "/") for s in subfiles]

    # Send warning if this is an empty directory
    if len(subfiles) == 0:
        print(colored("Warning: " + path + " is an empty directory.", "yellow"))
    return subfiles


def get_subfolds(data_path):
    """ This function will get every "subfold" that exists. """
    # Store paths in a dictionary
    subfolds = {}

    # Get the test subjects
    test_subject_paths = get_subfiles(data_path)

    # For each subject, get the configurations
    for subject in test_subject_paths:
        config_paths = get_subfiles(subject)
        subject_id = subject.split('/')[-1].split('_')[-1]

        # For each model/config, find its contents
        for config in config_paths:

            # Check if the model has contents
            subfold_paths = get_subfiles(config)
            if subfold_paths:

                # Check that the dictionary contains the model/subject
                model_name = subfold_paths[0].split('/')[-1].split('_')[0]
                if model_name not in subfolds:
                    subfolds[model_name] = {}
                if subject_id not in subfolds[model_name]:
                    subfolds[model_name][subject_id] = []

                # Add to results
                subfolds[model_name][subject_id].extend(subfold_paths)

    # Return the directory-paths
    return subfolds


def get_subfolder_files(data_path, target_folder, isIndex=None, getValidation=False, getTesting=False, isCSV=True):
    """ This function will get a set of files from each "subfold" contained within a particular target directory. """
    # Will return a dictionary of arrays separated by model
    target_subfolder_files = {}

    # Get the existing subfolds
    subfolds = get_subfolds(data_path)

    # Iterate through the subfolds
    for model_name in subfolds:
        target_subfolder_files[model_name] = {}
        for subject_id in subfolds[model_name]:
            target_subfolder_files[model_name][subject_id] = []

            # Each subfold should contain the target subfolder. Try to find it.
            for subfold in subfolds[model_name][subject_id]:
                subfiles = get_subfiles(subfold, return_full_path=False)

                # Check if the target folder is in the list
                if target_folder not in subfiles:
                    print(colored("Warning: " + target_folder + " not detected in " + subfold, "yellow"))
                    continue

                # Get the items within the target folder
                full_target_path = os.path.join(subfold, target_folder)
                target_paths = get_subfiles(full_target_path)

                # Make sure these are CSV files
                if isCSV:
                    for file in target_paths:
                        if not file.endswith(".csv"):
                            target_paths.remove(file)

                # Get only 'val' or 'test' files if specified
                if getValidation:
                    for file in target_paths:
                        if "_test true label index" in file.split("/")[-1] or \
                                "_test predicted_index" in file.split("/")[-1]:
                            target_paths.remove(file)
                elif getTesting:
                    for file in target_paths:
                        if "_val true" in file.split("/")[-1] or "_val_predicted" in file.split("/")[-1]:
                            target_paths.remove(file)

                # Append specifically indexed or not results to list if needed
                if isIndex is None:
                    target_subfolder_files[model_name][subject_id].extend(target_paths)
                elif isIndex:
                    for file in target_paths:
                        if "index" in file:
                            target_subfolder_files[model_name][subject_id].append(file)
                else:
                    for file in target_paths:
                        if "index" not in file:
                            target_subfolder_files[model_name][subject_id].append(file)

    # Check if there were actually any files (if predicted index)
    if target_folder == 'prediction' and isIndex:
        test_config_key = list(target_subfolder_files.keys())[0]
        test_fold_key = list(target_subfolder_files[test_config_key].keys())[0]
        if not target_subfolder_files[test_config_key][test_fold_key]:
            print(colored('No indexed-predictions were found in the data files. Running formatter...', 'yellow'))
            reformat()
            target_subfolder_files = get_subfolder_files(data_path, target_folder, isIndex, getValidation, getTesting,
                                                         isCSV)
            if not target_subfolder_files[test_config_key][test_fold_key]:
                raise Exception(colored("Error: No prediction files were found. " +
                                        "Check the data inputs and run predicted_formatter.py", 'red'))

    # Return the target files
    return target_subfolder_files


def get_history_paths(data_path):
    """ This function will get every history file from each model-fold. """
    # Will return a dictionary of arrays separated by model
    histories = {}

    # Get the existing subfolds
    subfolds = get_subfolds(data_path)

    # Iterate through the subfolds
    for model_name in subfolds:
        histories[model_name] = {}
        for subject_id in subfolds[model_name]:
            histories[model_name][subject_id] = []

            # Each subfold should contain one history file. Try to find it.
            for subfold in subfolds[model_name][subject_id]:
                subfiles = get_subfiles(subfold)

                # Search for the history file
                missing = True
                for subfile in subfiles:
                    if subfile.endswith("history.csv"):
                        histories[model_name][subject_id].append(subfile)
                        missing = False
                        break

                # Warn that the target file was not detected
                if missing:
                    print(colored("Warning: a history file was not detected in " + subfold, "yellow"))
                    continue

    # Return results
    return histories
