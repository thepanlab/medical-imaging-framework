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



def get_files(data_path, target_folder, isIndex=None, getValidation=False, getTesting=False, isCSV=True):
    """ This function will get a set of files from each subfold contained within a particular target directory. """
    # Will return a dictionary of arrays separated by model
    target_files = {}

    # They cannot both be true
    if getValidation and getTesting:
        getValidation = False
        getTesting = False

    # Get the test subjects (Level 1)
    test_subject_paths = get_subfiles(data_path.replace('\\\\', '/'))

    # For each subject, get the configurations (Level 2)
    for subject in test_subject_paths:
        config_paths = get_subfiles(subject)
        subject_id = subject.split('/')[-1].split('_')[-1]

        # For each config, get its subfolds (Level 3)
        for config in config_paths:
            subfold_paths = get_subfiles(config)

            # Each subfold should contain the target folder. Try to find it.
            for subfold in subfold_paths:
                subfiles = get_subfiles(subfold, return_full_path=False)
                model_name = subfold.split('/')[-1].split('_')[0]

                # Check if the target folder is in the list
                if target_folder not in subfiles:
                    print(colored("Warning: " + target_folder + " not detected in " + subfold, "yellow"))
                    continue

                # Get the items within the target folder
                full_target_path = os.path.join(subfold, target_folder)
                target_paths = get_subfiles(full_target_path)

                # Check model-array exists
                if model_name not in target_files:
                    target_files[model_name] = {}

                # Check if subject-array exists
                if subject_id not in target_files[model_name]:
                    target_files[model_name][subject_id] = []

                # Make sure these are CSV files
                if isCSV:
                    for file in target_paths:
                        if not file.endswith(".csv"):
                            target_paths.remove(file)

                # Get only 'val' or 'test' files if specified
                if getValidation:
                    for file in target_paths:
                        if "_test true" in file.split("/")[-1] or "_test predicted" in file.split("/")[-1]:
                            target_paths.remove(file)
                elif getTesting:
                    for file in target_paths:
                        if "_val true" in file.split("/")[-1] or "_val_predicted" in file.split("/")[-1]:
                            target_paths.remove(file)

                # Append specifically indexed or not results to list if needed
                if isIndex is None:
                    target_files[model_name][subject_id].extend(target_paths)
                elif isIndex:
                    for file in target_paths:
                        if "index" in file:
                            target_files[model_name][subject_id].append(file)
                else:
                    for file in target_paths:
                        if "index" not in file:
                            target_files[model_name][subject_id].append(file)
                        
    # Return the target files
    return target_files



def get_subfolds(data_path):
    """ This function will get every fold-path that exists. """
    # Store paths in a simple array
    subfolds = []

    # Get the test subjects
    test_subject_paths = get_subfiles(data_path)

    # For each subject, get the configurations
    for subject in test_subject_paths:
        config_paths = get_subfiles(subject)

        # For each config, get its subfolds and add to list
        for config in config_paths:
            subfold_paths = get_subfiles(config)
            subfolds.extend(subfold_paths)

    # Return the directory-paths
    return subfolds
