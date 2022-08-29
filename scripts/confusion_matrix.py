from sklearn.metrics import confusion_matrix
from termcolor import colored
import pandas as pd
import numpy as np
import regex as re
import ultraimport
import math
import os 

# Imports a module from a level below.
# If moved to same level, use "import get_config".
get_config = ultraimport('/home/jshaw/medical-imaging-framework/scripts/graphing/get_config.py')

# TODO: include preprocessing of the prob values (can be found in E3b-Processing predictions first few cells)



def create_confusion_matrix(true_vals, pred_vals, results_path, name, labels):
    """ Creates confusion matrix and saves as a csv in the results directory.  """ 
    # Check the file is valid
    file_name = re.sub('.csv', '', name)
    if len(true_vals) != len(pred_vals):
        raise Exception('The length of true and predicted values are not equal.')

    # Create the matrix
    conf_matrix = confusion_matrix(true_vals, pred_vals)
    header = pd.MultiIndex.from_product([["Predicted"], labels], names=['', 'True'])
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=header, index=labels)
    conf_matrix_df.to_csv(os.path.join(results_path, file_name) + '_conf_matrix.csv')
    print(colored("Confusion matrix created for " + name, 'green')) 
    return conf_matrix



def average_conf_matrices(matrices_list, results_path, name, labels):
    """ Takes a list of confusion matrices already computed and averages their results. Averages matrix is stored in the results directory.   """
    # Compute the average using a list of computed matrices
    avg_matrix = np.mean(matrices_list, axis=0)
    num_matrices = len(matrices_list)
    header = pd.MultiIndex.from_product([["Predicted"], labels], names=['', 'True'])
    avg_matrix_df = pd.DataFrame(avg_matrix, columns=header, index=labels)
    avg_matrix_df.to_csv(os.path.join(results_path, name) + '_avg_conf_matrix.csv')
    print(colored("Average confusion matrix created for " + name, 'green')) 

    # Compute the standard error of the matrices
    stderr_matrix = np.std(matrices_list, axis=0, ddof=0)/(math.sqrt(num_matrices))
    stderr_matrix_df = pd.DataFrame(stderr_matrix, columns=header, index=labels)
    stderr_matrix_df.to_csv(os.path.join(results_path, name) + '_stderr_conf_matrix.csv')
    print(colored("Standard error confusion matrix created for " + name, 'green'))



def get_data(pred_path, true_path):
    """ Reads in the labels and predictions from CSV """
    # Read CSV file
    pred = pd.read_csv(pred_path, header=None).to_numpy()
    true = pd.read_csv(true_path, header=None).to_numpy()

    # Get shapes
    pred_rows, pred_cols = pred.shape
    true_rows, true_cols = true.shape

    # Make the number of rows equal, in case uneven [[ TODO: Keep in with official data? ]]
    if pred_rows > true_rows:
        pred = pred[:true_rows, :]
    elif pred_rows < true_rows:
        true = true[:pred_rows, :]

    # Return true and predicted values
    return true, pred



""" The main program """
def main(config=None):
    # Obtaining dictionary of configurations from json file
    if config is None:
        config = get_config.parse_json('confusion_matrix_config.json')

    # Check that the output path exists
    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Obtain needed labels and predictions
    pred_path = config['pred_path']
    true_path = config['true_path']
    if not os.path.exists(pred_path):
        raise Exception(colored("Error: The prediction path is not valid!: " + pred_path, 'red'))
    if not os.path.exists(true_path):
        raise Exception(colored("Error: The true-value path is not valid!: " + true_path, 'red'))
    true_val, pred_val = get_data(pred_path, true_path)


    # Create a confusion matrix for the given files
    matrices_list = []
    conf_matrix = create_confusion_matrix(true_val , pred_val, output_path, config['output_file_prefix'], config['label_types'])
    matrices_list.append(conf_matrix)
    average_conf_matrices(matrices_list, output_path, config['output_file_prefix'], config['label_types'])



""" Executes the program """
if __name__ == "__main__":
    main()