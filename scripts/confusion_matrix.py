from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os 
import argparse
import json
import math

#TO DO: 
# label list include in config file - Done
# Pandas Column header - waiting
# File iteration - Done

def create_confusion_matrix(true_vals, pred_vals, results_path, name, labels):
    """ Creates confusion matrix and saves as a csv in the results directory.  
    """ 

    if len(true_vals) != len(pred_vals):
        raise Exception('The length of true and predicted values are not equal.')

    else:
        conf_matrix = confusion_matrix(true_vals, pred_vals)

        conf_matrix_df = pd.DataFrame(conf_matrix, columns=labels, index=labels)

        conf_matrix_df.to_csv(os.path.join(results_path, name) + '_conf_matrix.csv')
    
    print("Confusion matrix created for " + name) 

    return conf_matrix


def average_conf_matrices(matrices_list, results_path, name, labels):
    """ Takes a list of confusion matrices already computed and averages their results. Averages matrix is stored in the results directory.   
    """ 

    avg_matrix = np.mean(matrices_list, axis=0)

    num_matrices = len(matrices_list)

    avg_matrix_df = pd.DataFrame(avg_matrix, columns=labels, index=labels)

    avg_matrix_df.to_csv(os.path.join(results_path, name) + '_avg_conf_matrix.csv')

    print("Average confusion matrix created for " + name) 

    stderr_matrix = np.std(matrices_list, axis=0, ddof=0)/(math.sqrt(num_matrices))

    stderr_matrix_df = pd.DataFrame(stderr_matrix, columns=labels, index=labels)

    stderr_matrix_df.to_csv(os.path.join(results_path, name) + '_stderr_conf_matrix.csv')

    print("Standard error confusion matrix created for " + name)
    

def get_data(path):
    """ Gets labels and predictions from csv file.   
    """ 

    vals = pd.read_csv(path)
    
    y_val = vals['y_val']
    pred_val = vals['pred_val']

    return y_val, pred_val

def get_config_file(args):
    """Gets configuration from the 'conf_matrix_config.json' file. 
    """

    with open(args.load_json) as config_file:
        results_config = json.load(config_file)

    #Returns configurations as a dictionary
    return results_config

def main():

    #Obtaining dictionary of configurations from json file
    parser = argparse.ArgumentParser()
    parser.add_argument('load_json', help='Load settings from file in json format.')
    args = parser.parse_args()
    configs = get_config_file(args)

    #Establishing needed directories
    input_file_path = configs['input_path']
    output_path = configs['output_path']
    label_types = configs['label_types']

    #Creating list of files needed to process
    list_files = os.listdir(input_file_path)

    matrices_list = []

    for file in list_files:

        #Obtaining needed labels and predictions
        y_val, pred_val = get_data(os.path.join(input_file_path, file))

        conf_matrix = create_confusion_matrix(y_val , pred_val, output_path, file, label_types)

        matrices_list.append(conf_matrix)
    
    average_conf_matrices(matrices_list, output_path, "test", label_types)

if __name__ == "__main__":
    main()