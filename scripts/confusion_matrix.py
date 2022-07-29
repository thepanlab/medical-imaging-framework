from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os 
import argparse
import json
import math

#TO DO: 
#Add assert as safe guards
#Finish adding std err portion 

def create_confusion_matrix(true_vals, pred_vals, results_path, name):
    """ Creates confusion matrix and saves as a csv in the results directory.  
    """ 

    conf_matrix = confusion_matrix(true_vals, pred_vals)

    conf_matrix_df = pd.DataFrame(conf_matrix, columns=['fat', 'ligament', 'flavum', 'empty', 'spinalcord'], index=['fat', 'ligament', 'flavum', 'empty', 'spinalcord'])

    conf_matrix_df.to_csv(results_path + name + '_conf_matrix.csv')

def average_conf_matrices(matrices_list, results_path, name):
    """ Takes a list of confusion matrices already computed and averages their results. Averages matrix is stored in the results directory.   
    """ 

    avg_matrix = np.mean(matrices_list, axis=0)

    num_matrices = len(matrices_list)

    std_err_matrix = np.std(matrices_list, axis=0, ddof=0)/(math.sqrt(num_matrices))

    avg_matrix_df = pd.DataFrame(avg_matrix, columns=['fat', 'ligament', 'flavum', 'empty', 'spinalcord'], index=['fat', 'ligament', 'flavum', 'empty', 'spinalcord'])

    avg_matrix_df.to_csv(results_path + name + '_avg_conf_matrix.csv')
    

def get_data(path):
    """ Gets labels and predictions from csv file.   
    """ 

    vals = pd.read_csv(path + '.csv')
    
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
    file_name = 'test_df_cm'

    #Obtaining needed labels and predictions
    y_val, pred_val = get_data(input_file_path + file_name)

    create_confusion_matrix(y_val , pred_val, output_path, file_name)

if __name__ == "__main__":
    main()