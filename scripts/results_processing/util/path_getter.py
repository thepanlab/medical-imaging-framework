from .predicted_formatter.predicted_formatter import main as reformat
from termcolor import colored
import regex as re
import sys
import os


sys.setrecursionlimit(10**6)


def is_outer_loop(data_path):
    """ Checks if a given data path is of an inner or outer loop.

    Args:
        data_path (str): The path to a data directory.

    Raises:
        Exception: If a data path has no files.

    Returns:
        bool: If a directory is of an outer-loop structure.
    """
    # Get the test-fold directories
    test_subject_paths = get_subfiles(data_path)
    if not test_subject_paths:
        raise Exception(colored(f'Error: no files found in {data_path}', 'r'))
    
    # Get the children of the first test-fold directory
    subject_paths = get_subfiles(test_subject_paths[0], return_full_path=True)
    if not subject_paths:
        raise Exception(colored(f'Error: no files found in {test_subject_paths[0]}', 'r'))
    
    # The inner loop will have 'config' in its directories
    config_count = [p for p in subject_paths if 'config' in p.split('/')[-1]]
    config_count = [p for p in config_count if os.path.isdir(p)]
    if len(config_count) > 1:
        print(colored("Data detected as inner loop.", 'yellow'))
        return False
    print(colored("Data detected as outer loop.", 'yellow'))
    return True
        


def get_subfiles(path, return_full_path=True):
    """ This will get a list of everything contained within some particular directory; by model and subject.

    Args:
        path (str): The path to a data directory.
        return_full_path (bool, optional): Whether to return just the file name or the full path. Defaults to True.

    Raises:
        Exception: If no directory exists.

    Returns:
        list: A list of file paths.
    """
    # Make this path have all forward-slashes, to make usable with Windows machines
    path = path.replace("\\", "/")

    # Check if the path exists
    if not os.path.exists(path):
        raise Exception(colored(f"Error: The directory does not exist! - {path}", "red"))
    
    # If a file
    if not os.path.isdir(path):
        raise Exception(colored(f"Error: Not a directory, but it should be - {path}\n\t Check your data's directory structure.", "red"))

    # Get and return items
    subfiles = os.listdir(path)
    if return_full_path:
        subfiles = [os.path.join(path, s).replace("\\", "/") for s in subfiles]

    # Send warning if this is an empty directory
    if len(subfiles) == 0:
        print(colored("Warning: " + path + " is an empty directory.", "yellow"))
    return subfiles


def get_subfolds(data_path):
    """ This function will get every "subfold" that exists.

    Args:
        data_path (str): The path to a data directory.

    Returns:
        dict: Of paths organized by structure.
    """
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
            
            # Add check for if this contains info for 1 fold.
            info = {'file_name': False, 'model': False, 'prediction': False, 'true_label': False}
            for path in subfold_paths:
                subpath = path.split('/')[-1]
                if subpath in info:
                    info[subpath] = True
                    
            if subfold_paths:

                # Check that the dictionary contains the model/subject
                model_name = subfold_paths[0].split('/')[-1].split('_')[0]
                if model_name not in subfolds:
                    subfolds[model_name] = {}
                if subject_id not in subfolds[model_name]:
                    subfolds[model_name][subject_id] = []

                # Add to results
                if False in list(info.values()):
                    subfolds[model_name][subject_id].extend(subfold_paths)
                else:
                    subfolds[model_name][subject_id].append(config)

    # Return the directory-paths
    return subfolds


def get_subfolder_files(data_path, target_folder, isIndex=None, getValidation=False, getTesting=False, isCSV=True, returnIsOuter=False, isOuter=None):
    """ This function will get a set of files from each "subfold" contained within a particular target directory.

    Args:
        data_path (str): _description_
        target_folder (str): _description_
        isIndex (bool, optional): Whether to return the indexed or probability results. Defaults to None to return both.
        getValidation (bool, optional): Gets validation files only. Defaults to False.
        getTesting (bool, optional): Gets test files only. Defaults to False.
        isCSV (bool, optional): Check if the file should be a CSV file. Defaults to True.
        returnIsOuter (bool, optional): To return whether the data is determined to be in outer loop format.
        isOuter (bool, optional): A specification to whether this data path is the outer loop.

    Raises:
        Exception: If no files are found.

    Returns:
        dict: Of paths organized by structure.
        bool: True if the directory is an outer loop.
    """
    # Will return a dictionary of arrays separated by model
    target_subfolder_files = {}
    
    # Check if outer loop
    if isOuter is None:
        is_outer = is_outer_loop(data_path)
    else:
        is_outer = isOuter

    # Change both 'gets' to true if false
    if not getValidation and not getTesting:
        getValidation = True
        getTesting = True

    # Get the existing subfolds
    subfolds = get_subfolds(data_path)

    # Iterate through the subfolds
    for model_name in subfolds:
        target_subfolder_files[model_name] = {}
        for subject_id in subfolds[model_name]:
            target_subfolder_files[model_name][subject_id] = []

            # Each subfold should contain the target subfolder. Try to find it.
            for subfold in subfolds[model_name][subject_id]:
                
                # Make sure the directory is there...
                if target_folder in [f.split('/')[-1] for f in subfolds[model_name]] or \
                 target_folder in [f.split('/')[-1] for f in subfolds[model_name][subject_id]]:
                    raise Exception(colored("Error: Please make sure your data is in the correct format!\n\t" +
                            "The levels should be model->subject->fold. You have less than this.", 'red'))
                    
                # Get subfiles
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
                temp_paths = []
                if not is_outer:
                    if getValidation:
                        for file in target_paths:
                            if re.search('_test_.*_val_.*_val', file.split('/')[-1]) is not None:
                                temp_paths.append(file)
                    elif getTesting:
                        for file in target_paths:
                            if re.search('_test_.*_val_.*_test_', file.split('/')[-1]) is not None:
                                temp_paths.append(file)
                else: # TODO
                    for file in target_paths:
                        if re.search('.*_test_.*', file.split('/')[-1]) is not None:
                            temp_paths.append(file)
                target_paths = temp_paths
                
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
            reformat(data_path, is_outer=isOuter)
            target_subfolder_files = get_subfolder_files(data_path, target_folder, isIndex, getValidation, getTesting, isCSV, isOuter)
            if not target_subfolder_files[test_config_key][test_fold_key]:
                raise Exception(colored("Error: No prediction files were found. " +
                                        "Check the data inputs and run predicted_formatter.py", 'red'))

    # Return the target files
    if returnIsOuter:
        return target_subfolder_files, is_outer
    return target_subfolder_files


def get_history_paths(data_path):
    """ This function will get every history file from each model-fold.

    Args:
        data_path (str): Path to the data directory.

    Returns:
        dict: Of paths organized by structure.
    """
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


def get_config_indexes(data_path):
    """ Returns the config names and their index (E.x. config 1 -> resnet) 

    Args:
        data_path (str): Path to the data directory.

    Returns:
        dict: A model-index dictionary.
    """
    # Return a dict
    results = {}

    # The test folds
    test_subject_paths = get_subfiles(data_path)

    # For each subject, get the configurations
    for subject in test_subject_paths:
        config_paths = get_subfiles(subject)

        # For each model/config, find its contents
        for config in config_paths:
            config_id = config.split("/")[-1].split("_")[1]

            # Check if the model has contents
            subfold_paths = get_subfiles(config)
            if subfold_paths:
                # Check that the dictionary contains the model/subject
                model_name = subfold_paths[0].split('/')[-1].split('_')[0]
                results[model_name] = config_id
    return results