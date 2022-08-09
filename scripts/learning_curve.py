import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import json
import argparse
import regex as re

#TO DO: 
# Add progress message "lc produced" - Done
# Add verification for the needed files -Done
# remove box leave needed axes - Done
# Change title, make more specific (Subject #(after k) Learning curve) - Done

def lc_loss(path, data_frame, config, name):
    """ Creates the model loss plot using the defined configurations and saves it in the results directory.  
    """ 

    subject_num = get_subject_num(name)

    #Grabbing accuracy and val_accuracy data from passed dataframe
    loss = data_frame['loss']
    val_loss = data_frame['val_loss']

    #Establishing the plot's colors
    loss_line_color = config['loss_line_color']
    val_loss_line_color = config['val_loss_line_color']

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
    plt.title(subject_num.upper() + ' learning curve')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')

    #Removing top and right axis lines
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    #Saving plot to results directory and closing figure to avoid further plotting
    save_format = config['save_format']
    save_res = config['save_resolution']

    plt.savefig(os.path.join(path, name) + '_lc_loss' + '.' + save_format, format=save_format, dpi=save_res)
    plt.close()

    print("Loss Learning curve has been created for - " + name )

def lc_accuracy(path, data_frame, config, name):
    """ Creates the model accuracy plot using the defined configurations and saves it in the results directory.  
    """ 

    subject_num = get_subject_num(name)
    
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
    plt.title(subject_num.upper() + ' learning curve')
    plt.ylabel('accuracy', )
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')

    #Saving plot to results directory and closing plot
    save_format = config['save_format']
    save_res = config['save_resolution']

    plt.savefig(os.path.join(path, name) + '_lc_accuracy' + '.' + save_format, format=save_format, dpi=save_res)
    plt.close()

    print("Accuracy Learning curve has been created for - " + name )

def get_subject_num(file_name):
    try: 
        subject_search = re.search('(k[0-9])', file_name)
        subject_num = subject_search.captures()[0]
        return subject_num
    except: 
        raise Exception("File name does not contain k[0-9] format.")

def get_results_config(args):
    """Gets configuration information for the model accuracy and loss plots from the 'results_config.json' file. 
    """

    with open(args.load_json) as config_file:
        results_config = json.load(config_file)

    #Returns configurations as a dictionary
    return results_config

def create_graphs(file_list, file_path, results_path, results_config):  

    #Lopping through each file and creating needed graphs
    for file in file_list:

        #Reading in CSV file into a dataframe
        results_df = pd.read_csv(os.path.join(file_path, file),index_col = 0)

        file_name = re.sub('.csv', '', file)

        #Creating accuracy and loss plots
        lc_accuracy(results_path, results_df, results_config, file_name)
        lc_loss(results_path, results_df, results_config, file_name)

def file_verification(files_list):
    verified_list = []
    for file in files_list:
        history_check = re.search("history", file)
        if history_check != None:
            verified_list.append(file)
        else:
            pass
    return verified_list

def main():

    #Obtaining dictionary of configurations from json file
    parser = argparse.ArgumentParser()
    parser.add_argument('load_json', help='Load settings from file in json format.')
    args = parser.parse_args()
    results_config = get_results_config(args)

    file_path = results_config['input_path']
    results_path = results_config['output_path']

    #Creating list of files needed to process
    list_files = os.listdir(file_path)

    verified_list_files = file_verification(list_files)

    create_graphs(verified_list_files, file_path, results_path, results_config)

if __name__ == "__main__":
    main()