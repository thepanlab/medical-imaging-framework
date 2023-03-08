from util.get_config import parse_json
from termcolor import colored
from util import path_getter
import pandas as pd
import math
import os


def count_epochs(history_paths, is_outer):
    """ Reads in the history paths and finds the ideal loss.

    Args:
        history_paths (dict): A dictionary of file locations.
        is_outer (bool): If this data is from the outer loop or not.

    Raises:
        Exception: When no history files exist for some item.

    Returns:
        dict: A dictionary of minimum epoch losses.
    """
    # Store output dataframes by model
    model_dfs = {}

    # Each model will have its own dictionary, a later dataframe
    for model in history_paths:
        model_dfs[model] = {}

        # Every subject is the kth test set (row), find the column values
        for row in history_paths[model]:
            row_name = row
            model_dfs[model][row_name] = {}

            # Read in the data at this path
            for path in history_paths[model][row]:
                # Check if the fold has files to read from
                if not path:
                    raise Exception(colored("Error: model '" + model + "' and target '" + target_id
                                            + "' had no history file detected.", 'red'))

                # Read the file for this column/subject, get number of rows (epochs)
                data = pd.read_csv(path)
                min_index = -1
                min_epoch = float("inf")
                if is_outer:
                    key = 'loss'
                else:
                    key = 'val_loss'
                for row in data[key].index:
                    if data[key][row] < min_epoch:
                        min_epoch = data[key][row]
                        min_index = row

                # Add the epoch with the lowest loss the model's dataframe 
                col_name = path.split("/")[-2].split("_")[-1]
                model_dfs[model][row_name][col_name] = min_index + 1

    # Return a dictionary of counts
    return model_dfs


def print_counts(epochs, output_path, config_nums, is_outer):
    """ This will output a CSV of the epoch counts.

    Args:
        epochs (dict): Dictionary of epochs with minimum loss.
        output_path (str): Path to write files into.
        config_nums (dict): The configuration indexes of the data.
        is_outer (bool): If this data is from the outer loop.
    """
    # Create a new dataframe to output
    col_names = ["test_fold", "config", "config_index", "val_fold", "epochs"]
    df = pd.DataFrame(columns=col_names)

    # Re-format data to match the columns above
    configs = list(epochs.keys())
    for testing_fold_index in range(len(epochs[configs[0]])):
        for config in configs:
            testing_fold = list(epochs[config].keys())[testing_fold_index]
            for validation_fold in epochs[config][testing_fold]:

                # Each row should contain the given columns
                df = df.append({
                    col_names[0]: testing_fold,
                    col_names[1]: config,
                    col_names[2]: config_nums[config],
                    col_names[3]: validation_fold,
                    col_names[4]: epochs[config][testing_fold][validation_fold]
                }, ignore_index=True)

    # Print to file
    if is_outer:
        file_name = 'epochs_outer.csv'
    else:
        file_name = 'epochs_inner.csv'
    df = df.sort_values(by=[col_names[0], col_names[2], col_names[1]], ascending=True)
    df.to_csv(os.path.join(output_path, file_name), index=False)
    print(colored('Successfully printed epoch results to: ' + file_name, 'green'))


def print_stderr(epochs, output_path, config_nums):
    """ This will output a CSV of the average epoch standard errors.

    Args:
        epochs (dict): Dictionary of epochs with minimum loss.
        output_path (str): Path to write files into.
        config_nums (dict): The configuration indexes of the data.
    """
    # Create a new dataframe to output
    col_names = ["test_fold", "config", "config-index", "avg_epochs", "std_err"]
    df = pd.DataFrame(columns=col_names)

    # Re-format data to match the columns above
    for config in epochs:
        for test_fold in epochs[config]:

            # Count epochs, get mean
            epoch_mean = 0
            n_val_folds = len(epochs[config][test_fold])
            for validation_fold in epochs[config][test_fold]:
                epoch_mean += epochs[config][test_fold][validation_fold]
            epoch_mean = epoch_mean / n_val_folds

            #  Get standard deviation
            stdev = 0
            for validation_fold in epochs[config][test_fold]:
                stdev += (epochs[config][test_fold][validation_fold] - epoch_mean) ** 2
            stdev = math.sqrt(stdev / (n_val_folds - 1))

            # Each row should contain the given columns
            df = df.append({
                col_names[0]: test_fold,
                col_names[1]: config,
                col_names[2]: config_nums[config],
                col_names[3]: epoch_mean,
                col_names[4]: stdev / math.sqrt(n_val_folds)
            }, ignore_index=True)

    # Print to file
    file_name = 'epoch_inner_avg_stderr.csv'
    df = df.sort_values(by=[col_names[0], col_names[1]], ascending=True)
    df.to_csv(os.path.join(output_path, file_name), index=False)
    print(colored('Successfully printed epoch averages/stderrs to: ' + file_name, 'green'))


def main(config=None):
    """ The main body of the program """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('./results_processing/epoch_counting/epoch_counting_config.json')
    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])

    # Get the necessary input files
    history_paths = path_getter.get_history_paths(config['data_path'])
    is_outer = config['is_outer']

    # Count the number of epochs within every file
    epochs = count_epochs(history_paths, is_outer)

    # Get config nums (E.g config 1)
    config_nums = path_getter.get_config_indexes(config['data_path'])

    # Output the counts
    print_counts(epochs, config['output_path'], config_nums, is_outer)

    # Output the stderr
    if not is_outer:
        print_stderr(epochs, config['output_path'], config_nums)


if __name__ == "__main__":
    main()
