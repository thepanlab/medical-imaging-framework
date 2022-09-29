from util.get_config import parse_json
from termcolor import colored
from util import path_getter
from statistics import mode
import pandas as pd
import regex as re
import os


def get_input_matrices(matrices_path):
    """ Finds the existing configs, test folds, and validations folds of all matrices.

    Args:
        matrices_path (string): The path of the matrices' directory.

    Returns:
        dict: A dictionary of matrix-dataframes amd their prediction shapes, organized by config and testing fold.
    """
    # Get the confusion matrix paths that were created earlier
    all_paths = os.listdir(matrices_path)
    organized_paths = {}
    organized_shapes = {}

    # Separate by config
    for path in all_paths:
        config = path.split("/")[0].split('_')[0]
        if config not in organized_paths:
            organized_paths[config] = []
        organized_paths[config].append(path)

    # For each config, separate by testing fold
    for config in organized_paths:
        config_paths = organized_paths[config]
        organized_paths[config] = {}
        organized_shapes[config] = {}
        for path in config_paths:
            
            # Search for the test-fold name from the file name
            test_fold = re.search('_test_.*_val_.*_val', path).captures()[0].split("_")[2]
            if test_fold not in organized_paths[config]:
                organized_paths[config][test_fold] = {}
                organized_shapes[config][test_fold] = {}
                
            # Search for the val-fold name from the file name, read the csv
            val_fold = re.search('_test_.*_val_.*_val', path).captures()[0].split("_")[4]
            organized_paths[config][test_fold][val_fold] = pd.read_csv(os.path.join(matrices_path, path), header=[0, 1], index_col=[0, 1])

            # Search for the shape-value from the file name
            shape = int(re.search('index_.*_conf_matrix.csv', path).captures()[0].split("_")[1])
            organized_shapes[config][test_fold][val_fold] = shape

    # Return the dictionary of organized matrices
    return organized_paths, organized_shapes


def get_matrices_of_mode_shape(shapes, matrices):
    """ Finds the mode length of all predictions that exist within a data folder.
        Checks if matrices should be considered in the mean value.

    Args:
        shapes (dict): The shapes (prediction rows) of the corresponding confusion matrices.
        matrices (dict): A dictionary of the matrices.

    Returns:
        dict: A reduced dictionary of matrix-dataframes, organized by config and testing fold.
    """
    # Get the mode of ALL the prediction shapes
    shapes_mode = []
    for config in matrices:
        for test_fold in matrices[config]:
            for val_fold in matrices[config][test_fold]:
                shapes_mode.append(shapes[config][test_fold][val_fold])
    shapes_mode = mode(shapes_mode)

    # Remove matrices whose prediction value length do not match the mode
    for config in matrices:
        for test_fold in matrices[config]:
            
            # Each testing fold will have an array of coresponding validation matrices
            test_fold_matrices = []
            for val_fold in matrices[config][test_fold]:
                val_fold_shape = shapes[config][test_fold][val_fold]
                if val_fold_shape == shapes_mode:
                    test_fold_matrices.append(matrices[config][test_fold][val_fold])
                else:
                    print(colored(
                        f"Warning: Not including the validation fold {val_fold} in the mean of ({config}, {test_fold})." +
                        f"\n\tThe predictions expected to have {shapes_mode} rows, but got {val_fold_shape} rows.\n",
                        "yellow"))
            matrices[config][test_fold] = test_fold_matrices
    return matrices



def get_mean_matrices(matrices, shapes, output_path, labels):
    """ This function gets the mean confusion matrix of every inner loop.

    Args:
        matrices (dict): A dictionary of matrices organized by config and testing fold.
        shapes (dict): A dictionary of shapes organized by config and testing fold.
        output_path(string): The path to write the average matrices to.
        labels(list(str)): The labels of the matrix data.
    """
    # Check that the output folder exists.
    if not os.path.exists(output_path): os.makedirs(output_path)

    # Get the valid matrices
    all_valid_matrices = get_matrices_of_mode_shape(shapes, matrices)

    # The names of the rows/columns of every output matrix
    index_labels = [["Truth"] * len(labels), labels]
    columns_labels = [["Predicted"] * len(labels), labels]

    # Get the mean of each test fold + configuration
    for config in matrices:
        for test_fold in matrices[config]:

            # Check shape is the mode for each validation fold
            valid_matrices = all_valid_matrices[config][test_fold]

            # Check if length is valid for finding mean/stderr
            n_items = len(valid_matrices)
            if not n_items > 1:
                print(colored(f"Warning: Mean calculation skipped for testing fold {test_fold} in {config}."
                              + " Must have multiple validation folds.\n", 'yellow'))
                continue

            # The mean of confusion matrices and the weighted matrices
            matrix_avg = pd.DataFrame(0.0, columns=labels, index=labels)
            matrix_weighted_avg = pd.DataFrame(0.0, columns=labels, index=labels)
            matrix_avg.index = matrix_weighted_avg.index = index_labels
            matrix_avg.columns = matrix_weighted_avg.columns = columns_labels

            # The original weighted matrix values
            weighted_values = {}

            # Loop through every validation fold and sum the totals and weighted totals
            for val_index in range(len(valid_matrices)):
                val_fold = valid_matrices[val_index]
                weighted_values[val_index] = pd.DataFrame(0, columns=labels, index=labels)
                weighted_values[val_index].index = index_labels
                weighted_values[val_index].columns = columns_labels
                for row in labels:

                    # Count the row total and add column-row items to total
                    row_total = 0
                    for col in labels:
                        item = val_fold["Predicted"][col]["Truth"][row]
                        matrix_avg["Predicted"][col]["Truth"][row] += item
                        row_total += item

                    # Add to the weighted total
                    for col in labels:
                        item = val_fold["Predicted"][col]["Truth"][row]
                        weighted_item = item / row_total
                        weighted_values[val_index]["Predicted"][col]["Truth"][row] = weighted_item
                        matrix_weighted_avg["Predicted"][col]["Truth"][row] += weighted_item
                        
            # Divide the mean-sums by the length
            for row in labels:
                for col in labels:
                    matrix_avg["Predicted"][col]["Truth"][row] /= n_items
                    matrix_weighted_avg["Predicted"][col]["Truth"][row] /= n_items

            # The standard error of the mean and weighted mean
            matrix_err = pd.DataFrame(0.0, columns=labels, index=labels)
            matrix_weighted_err = pd.DataFrame(0.0, columns=labels, index=labels)
            matrix_err.index = matrix_weighted_err.index = index_labels
            matrix_err.columns = matrix_weighted_err.columns = columns_labels

            # Sum the differences between the true and mean values squared. Divide by the num matrices minus one.
            for val_index in range(len(valid_matrices)):
                val_fold = valid_matrices[val_index]
                for row in labels:
                    for col in labels:

                        #print(colored(matrix_avg["Predicted"][col]["Truth"][row]/ n_items, 'red'))

                        # Sum the (true - mean) ^ 2
                        true = val_fold["Predicted"][col]["Truth"][row]
                        mean = matrix_avg["Predicted"][col]["Truth"][row]
                        matrix_err["Predicted"][col]["Truth"][row] += (true - mean) ** 2

                        # Sum the (true_weighted - mean_weighted) ^ 2
                        true_weighted = weighted_values[val_index]["Predicted"][col]["Truth"][row]
                        mean_weighted = matrix_weighted_avg["Predicted"][col]["Truth"][row]
                        matrix_weighted_err["Predicted"][col]["Truth"][row] += (true_weighted - mean_weighted) ** 2

            # Divide the error-sums by n_items - 1
            for row in labels:
                for col in labels:
                    matrix_err["Predicted"][col]["Truth"][row] /= n_items - 1
                    matrix_weighted_err["Predicted"][col]["Truth"][row] /= n_items - 1

            # Output ALL matrices
            output_folder = os.path.join(output_path, f'{config}_{test_fold}/',)
            if not os.path.exists(output_folder): os.makedirs(output_folder)
            matrix_avg.round(2).to_csv(os.path.join(
                output_folder, f'{config}_{test_fold}_conf_matrix_mean.csv'))
            matrix_weighted_avg.round(2).to_csv(os.path.join(
                output_folder, f'{config}_{test_fold}_conf_matrix_mean_weighted.csv'))
            matrix_err.round(2).to_csv(os.path.join(
                output_folder, f'{config}_{test_fold}_conf_matrix_mean_stderr.csv'))
            matrix_weighted_err.round(2).to_csv(os.path.join(
                output_folder, f'{config}_{test_fold}_conf_matrix_mean_stderr_weighted.csv'))
            print(colored(f"Mean confusion matrix results created for {test_fold} in {config} \n", 'green'))


def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json(os.path.abspath('confusion_matrix_many_means_config.json'))

    # Read in the matrices to average
    matrices, shapes = get_input_matrices(config['matrices_path'])

    # Average the matrices
    get_mean_matrices(matrices, shapes, config['means_output_path'], config['label_types'])


if __name__ == "__main__":
    """ Executes Program. """
    main()
