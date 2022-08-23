from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import regex as re
import argparse
import json
import math
import os 

#TO DO: 
# include preprocessing of the prob values (can be found in E3b-Processing predictions first few cells)



""" Creates confusion matrix and saves as a csv in the results directory.  """ 
def create_confusion_matrix(true_vals, pred_vals, results_path, name, labels):
    
    # Check the file is valid
    file_name = re.sub('.csv', '', name)
    if len(true_vals) != len(pred_vals):
        raise Exception('The length of true and predicted values are not equal.')

    # Create the matrix
    else:
        conf_matrix = confusion_matrix(true_vals, pred_vals)
        header = pd.MultiIndex.from_product([["Predicted"], labels], names=['', 'True'])
        conf_matrix_df = pd.DataFrame(conf_matrix, columns=header, index=labels)
        conf_matrix_df.to_csv(os.path.join(results_path, file_name) + '_conf_matrix.csv')
    print("Confusion matrix created for " + name) 
    return conf_matrix



""" Takes a list of confusion matrices already computed and averages their results. Averages matrix is stored in the results directory.   """
def average_conf_matrices(matrices_list, results_path, name, labels):
 
    # Compute the average using a list of computed matrices
    avg_matrix = np.mean(matrices_list, axis=0)
    num_matrices = len(matrices_list)
    header = pd.MultiIndex.from_product([["Predicted"], labels], names=['', 'True'])
    avg_matrix_df = pd.DataFrame(avg_matrix, columns=header, index=labels)
    avg_matrix_df.to_csv(os.path.join(results_path, name) + '_avg_conf_matrix.csv')
    print("Average confusion matrix created for " + name) 

    # Compute the standard error of the matrices
    stderr_matrix = np.std(matrices_list, axis=0, ddof=0)/(math.sqrt(num_matrices))
    stderr_matrix_df = pd.DataFrame(stderr_matrix, columns=header, index=labels)
    stderr_matrix_df.to_csv(os.path.join(results_path, name) + '_stderr_conf_matrix.csv')
    print("Standard error confusion matrix created for " + name)



""" Reads in the labels and predictions from CSV """
def get_data(pred_path, true_path):

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



""" Reads in the configuration from a JSON file """
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-j', '--json', '--load_json',
        help='Load settings from a JSON file.',
        required=False,
        default='confusion_matrix_config.json'
    )
    args = parser.parse_args()
    with open(args.json) as config_file:
        roc_config = json.load(config_file)
    return roc_config



""" The main program """
def main():

    # Obtain Configurations from json file
    cm_config = get_config()

    # Check that the output path exists
    output_path = cm_config['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Obtain needed labels and predictions
    pred_path = cm_config['pred_path']
    true_path = cm_config['true_path']
    if not os.path.exists(pred_path):
        raise Exception("Error: The prediction path is not valid!: " + pred_path)
    if not os.path.exists(true_path):
        raise Exception("Error: The true-value path is not valid!: " + true_path)
    true_val, pred_val = get_data(pred_path, true_path)


    # Create a confusion matrix for the given files
    matrices_list = []
    conf_matrix = create_confusion_matrix(true_val , pred_val, output_path, cm_config['output_file_prefix'], cm_config['label_types'])
    matrices_list.append(conf_matrix)
    average_conf_matrices(matrices_list, output_path, cm_config['output_file_prefix'], cm_config['label_types'])



""" Executes the program """
if __name__ == "__main__":
    main()