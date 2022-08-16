import json
import argparse
import os

"""
Learning curve: history files -> plots
Confusion matrix: true labels and prediction values -> csv files
Roc_curve: true labels and probabilities -> plots

Tasks: 
-determine number of loops 
    - for each loop done:
        - Proccess each test 
            - learning curve
                - grab history file
            - Confusion matrix 
                - grab true labels 
                - grab prediction values 
            - roc curve
                - grab true vals
                - grab predicted values
        - Process each val
            - learning curve
                - grab history file
            - Confusion matrix 
                - grab true labels 
                - grab prediction values 
            - roc curve
                - grab true vals
                - grab predicted values

"""

def get_config(args):
    """Gets configuration information for the model accuracy and loss plots from the 'output_config.json' file. 
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
    output_config = get_config(args)

    #Grabbing model directory
    model_path = output_config['input_path']

    #List of all loops 
    list_loops = os.listdir(model_path)

    #For each loop done, grab each neccessary file for ouput processing
    #Since there should be a standard structure of the results, this should work for each model folder
    for fold in list_loops:
        
        folder_path = os.path.join(model_path, fold)

        #Directory for history file to be used to creat the learning curve
        history_file = os.path.join(folder_path, fold + '_history.csv')

        #Prediction folder path 
        pred_path = os.path.join(folder_path, "prediction")

        #Name of test prediction file and path
        test_pred_filename = fold + "_test_predicted.csv"
        test_pred_file = os.path.join(pred_path, test_pred_filename)

        #Name of val prediction file and path
        val_pred_filename = fold + "_val_predicted.csv"
        val_pred_file = os.path.join(pred_path, val_pred_filename)

        #True_label folder path
        true_label_path = os.path.join(folder_path, "true_label")

        #Name of test true label file and path
        test_true_filename = fold + "_test true label.csv"
        test_true_file = os.path.join(true_label_path, test_true_filename)

        #Name of val true label file and path
        val_true_filename = fold + "_val true label.csv"
        val_true_file = os.path.join(true_label_path, val_true_filename)

if __name__ == "__main__":
    main()