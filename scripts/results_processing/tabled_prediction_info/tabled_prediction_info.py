from util.get_config import parse_json
from termcolor import colored
from util import path_getter
import pandas as pd
import regex as re
import os


def find_directories(config):
    """ Finds the directories for every input needed to make graphs.

    Args:
        config (dict): The program configuration.

    Returns:
        dict: Two dictionaries of prediction and truth paths.
    """
    # Get the paths of every prediction and true CSV, as well as the fold-names
    pred_paths = path_getter.get_subfolder_files(config['data_path'], "prediction", isIndex=True, getValidation=True, isOuter=config['is_outer'])
    if config['use_true_labels']:
        true_paths = path_getter.get_subfolder_files(config['data_path'], "true_label", isIndex=True, getValidation=True, isOuter=config['is_outer'])
        return pred_paths, true_paths
    else:
        return pred_paths, None


def find_images(data_path):
    """ Finds the images within a directory.

    Args:
        data_path (str): Path to a directory.

    Returns:
        dict: Dictionary of subfiles.
    """
    return path_getter.get_subfolder_files(data_path, "file_name")


def compare_values(output_path, pred_paths, true_paths, image_paths, label_types, is_outer):
    """ Compares the true and predicted values and gets other info.
    Args:
        output_path (str): Directory to output to.
        pred_paths (dict): Paths to the predictions.
        true_paths (dict): Paths to the true values. May be None if true labels aren't used.
        image_paths (dict): Paths to the images.
        label_types (dict): A list of labels and their index.
        is_outer (bool): If the data is from the outer loop.
    """
    # Explicitly declare a true label flag
    do_true = False if true_paths is None else True
    
    # Dataframe column order
    if is_outer:
        cols = ['test_subject',                'filename', 'true_label', 'true_label_index', 'pred_label', 'pred_label_index', 'match', 'filepath']
    else:
        cols = ['test_subject', 'val_subject', 'filename', 'true_label', 'true_label_index', 'pred_label', 'pred_label_index', 'match', 'filepath']
    if not do_true:
        cols.remove('true_label_index')
        cols.remove('true_label')
        cols.remove('match')
    
    # Iterate through each prediction-truth-filepath trio
    for config in pred_paths:
        results = pd.DataFrame(columns=cols)
        t_i = 0
        t_n = len(pred_paths[config].keys())
        print(colored(f"{config} began computing... {round((t_i/t_n)*100, 2)}%", 'cyan'), end='\r')
        for test_fold in pred_paths[config].keys():
            for pred_file in pred_paths[config][test_fold]:
                
                # If outer loop, only one file exists
                if is_outer:
                    if do_true:
                        true_file = true_paths[config][test_fold][0]
                    img_file = image_paths[config][test_fold][0]
                
                # If inner loop, verify the validation folds match
                else:
                    file_name = pred_file.split('/')[-1]
                    if "_val_" not in file_name:
                        raise ValueError(colored(f"Error: No validation subject detected in the prediction file name. Are you sure it is not of the outer loop?\n\t{file_name}", 'red'))
                    val_fold = re.search('_test_.*_val_.*_val', file_name).captures()[0].split("_")[4]
                    if do_true:
                        true_file = [t for t in true_paths[config][test_fold] if (f"_test_{test_fold}_val_{val_fold}_val" in t.split('/')[-1])][0]
                    img_file = [t for t in image_paths[config][test_fold] if (f"_test_{test_fold}_val_{val_fold}_val" in t.split('/')[-1])][0]
                
                # Read in the data
                pred_label_index = pd.read_csv(pred_file, header=None)[0].values.tolist()
                if do_true:
                        true_label_index = pd.read_csv(true_file, header=None)[0].values.tolist()
                filepaths = pd.read_csv(img_file, header=None)[0].values.tolist()
                filenames = [f.split('/')[-1] for f in filepaths]
                result_dict = {
                    'test_subject': [test_fold]*len(pred_label_index),
                    'true_label_index': None,
                    'pred_label_index': pred_label_index,
                    'filename': filenames,
                    'filepath': filepaths
                }
                if do_true:
                    result_dict['true_label_index'] = true_label_index  
                else:
                    del result_dict['true_label_index']
                if not is_outer:
                    result_dict['val_subject'] = [val_fold]*len(pred_label_index)
                file_result = pd.DataFrame(result_dict)
                
                # Compare the values to create new columns
                try:
                    file_result['pred_label'] = file_result.apply(lambda row: label_types[str(row.pred_label_index)], axis=1)
                except:
                    raise ValueError(colored("Error: key mismatch. Make sure your label dictionary values are correct.", 'red'))
                if do_true:
                    file_result['true_label'] = file_result.apply(lambda row: label_types[str(row.true_label_index)], axis=1)
                    file_result['match'] = file_result.apply(lambda row: row.pred_label_index == row.true_label_index, axis=1)
                results = pd.concat([results, file_result], ignore_index=True, sort=False)
            t_i += 1
            print(colored(f"{config} began computing... {round((t_i/t_n)*100, 2)}%", 'cyan'), end='\r')
        print(colored(f"{config} finished computing.        ", 'blue'))
        print_results(output_path, config, results)
    print(colored(f"Successfully calculated the prediction results.", 'green'))


def print_results(filepath, config, results):
    """ Prints the data to file.

    Args:
        filepath (str): Path to print to.
        config (str): The name of the configuration.
        results (pandas.DataFrame): Results of the predictions.
    """
    results.sort_values(by=['test_subject']).to_csv(os.path.join(filepath, f'{config}_prediction_info.csv'), index=False)
    print(colored(f"Successfully output the results for {config}.", 'magenta'))
                    

def main(config=None):
    """ The Main Program. """
    # Get program configuration and run using its contents
    if config is None:
        config = parse_json('./results_processing/tabled_prediction_info/tabled_prediction_info_config.json')
    pred_paths, true_paths = find_directories(config)
    print(colored(f"Successfully read in the prediction and truth paths.", 'green'))
    image_paths = find_images(config["data_path"])
    print(colored(f"Successfully read in the image paths.", 'green'))
    if not os.path.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    compare_values(config["output_path"], pred_paths, true_paths, image_paths, config["label_types"], config['is_outer'])


if __name__ == "__main__":
    """ Executes Program. """
    main()
