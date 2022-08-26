from termcolor import colored
import os 

"""
    Path Getter

    This file will find all of the model predictions, true-values, and other file names.
    This will expect a particular data-path structure.

    > data_path/
        > Test_subject_e1/                              # Level 1
            > config_1/                                 # Level 2
                > [some fold name]                      # Level 3
                    > prediction                                   <-- This can be a target folder
                        > [some fold name]_val_predicted.csv
                        > [some fold name]_val_predicted_index.csv
                        > ...
                    > true_label                                   <-- This can be a target folder
                        > [some fold name]_val true label.csv
                        > [some fold name]_val true label index.csv
                        > ...
                > ...
            > config_2/
            > config_3/
            > winner_model/
        > Test_subject_e2/
        ...

"""



def get_subfiles(path, return_full_path=True):
    """ This will get a list of everything contained within some particular directory. """
    # Check if the path exists and is a directory
    if not (os.path.exists(path) and os.path.isdir(path)):
        raise Exception(colored("Error: The directory does not exist! - " + path, "red"))
    
    # Get and return items
    subfiles = os.listdir(path)
    if return_full_path:
        subfiles = [os.path.join(path, s) for s in subfiles]

    # Send warning if this is an empty directory
    if len(subfiles) == 0:
        print(colored("Warning: " + path + " is an empty directory.", "yellow"))
    return subfiles



def get_files(data_path, target_folder, isIndex=None):
    """ This function will get a set of files from each fold contained within a particular target directory. """
    # Will return a dictionary of arrays separated by fold
    target_files = {}

    # Get the test subjects
    test_subject_paths = get_subfiles(data_path)

    # For each subject, get the configurations
    for subject in test_subject_paths:
        config_paths = get_subfiles(subject)

        # For each config, get its folds
        for config in config_paths:
            fold_paths = get_subfiles(config)

            # Each fold should contain the target folder. Try to find it.
            for fold in fold_paths:
                subfiles = get_subfiles(fold, return_full_path=False)
                fold_name = fold.split('/')[-1]

                # Check if the target folder is in the list
                if target_folder not in subfiles:
                    print(colored("Warning: " + target_folder + " not detected in " + fold, "yellow"))
                    continue

                # Get the items within the target folder
                full_target_path = os.path.join(fold, target_folder)
                target_paths = get_subfiles(full_target_path)

                # Check fold-array exists
                if fold_name not in target_files:
                    target_files[fold_name] = []

                # Append specifically indexed or not results to list if needed
                if isIndex is None:
                    target_files[fold_name].extend(target_paths)
                elif isIndex:
                    for file in target_paths:
                        if "index" in file:
                            target_files[fold_name].append(file)
                else:
                    for file in target_paths:
                        if "index" not in file:
                            target_files[fold_name].append(file)
                        
    # Return the target files
    return target_files
