from turtle import color
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import json

def lc_loss(path, data_frame, config):
    """ Creates the model loss plot using the defined configurations and saves it in the results directory.  
    """ 

    #Grabbing accuracy and val_accuracy data from passed dataframe
    loss = data_frame['loss']
    val_loss = data_frame['val_loss']

    #Establishing the plot's colors
    loss_line_color = config['loss_line_color']
    val_loss_line_color = config['val_loss_line_color']
    font_family = config['font_family']

    #Establishing the plot's font and font sizes
    font_family = config['font_family']
    label_font_size = config['label_font_size']
    title_font_size = config['title_font_size']

    #Setting the previously established parameters
    plt.rcParams['font.family'] = font_family
    plt.rc('axes', labelsize=label_font_size)
    plt.rc('axes', titlesize=title_font_size)

    #Creating plot
    plt.plot(loss, color=loss_line_color)
    plt.plot(val_loss, color=val_loss_line_color)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')

    #Saving plot to results directory
    plt.savefig(path + 'test_loss.jpg')

def lc_accuracy(path, data_frame, config):
    """ Creates the model accuracy plot using the defined configurations and saves it in the results directory.  
    """ 
    #Grabbing accuracy and val_accuracy data from passed dataframe
    acc = data_frame['accuracy']
    val_acc = data_frame['val_accuracy']

    #Establishing the plot's colors
    acc_line_color = config['acc_line_color']
    val_acc_line_color = config['val_acc_line_color']

    #Establishing the plot's font and font sizes
    font_family = config['font_family']
    label_font_size = config['label_font_size']
    title_font_size = config['title_font_size']

    #Setting the previously established parameters
    plt.rcParams['font.family'] = font_family
    plt.rc('axes', labelsize=label_font_size)
    plt.rc('axes', titlesize=title_font_size)

    #Creating plot
    plt.plot(acc, color=acc_line_color)
    plt.plot(val_acc, color=val_acc_line_color)
    plt.title('model accuracy')
    plt.ylabel('accuracy', )
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')

    #Saving plot to results directory
    plt.savefig(path + 'test_acc.jpg')

def get_results_config(path):
    """Gets configuration information for the model accuracy and loss plots from the 'results_config.json' file. 
    """

    with open(path + 'learningcurve_config.json') as config_file:
        results_config = json.load(config_file)

    #Returns configurations as a dictionary
    return results_config

def main():

    #Establishing needed directories
    dir_path = os.getcwd()
    data_path = dir_path + '/data/Sample Data/'
    current_path = dir_path + '/scripts/'
    results_path = dir_path + '/results/'
    file_name = 'model_1_k0_history.csv'
    file_path = data_path + file_name

    #Reading in CSV file into a dataframe
    results_df = pd.read_csv(file_path,index_col = 0)

    #Obtaining dictionary of configurations from json file
    results_config = get_results_config(current_path)

    #Creating accuracy and loss plots
    lc_accuracy(results_path, results_df, results_config)
    lc_loss(results_path, results_df, results_config)

if __name__ == "__main__":
    main()