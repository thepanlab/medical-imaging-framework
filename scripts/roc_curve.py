from sklearn.metrics import roc_curve, auc
import numpy as np
import os 
from matplotlib import pyplot as plt
from itertools import cycle
import argparse
import json
import pandas as pd
from sklearn.preprocessing import label_binarize

def create_roc_curve(true_vals, pred_vals, roc_config, file_name, results_path):

    #assuming true_vals is a numpy array
    classes = np.unique(true_vals)
    n_classes = len(classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    true_val_bin = label_binarize(true_vals, classes=classes)

    pred_val_bin = label_binarize(pred_vals, classes=classes)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_val_bin[:,i], pred_val_bin[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(true_val_bin.ravel(), pred_val_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = roc_config['line_width']
    l_types = roc_config['label_types']
    colors = cycle(roc_config['line_colors'])
    save_format = roc_config['save_format']
    save_res = roc_config['save_resolution']

    for i, type_name, color in zip(classes, l_types, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='{0} (AUC = {1:0.2f})'
                ''.format(type_name, roc_auc[i]))

        
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver operating characteristic(ROC) multi-class Testing')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_path, file_name) + '.' +save_format, dpi=save_res)
    plt.close()

def get_data(path):
    """ Gets labels and predictions from csv file.   
    """ 
    path = "/home/ecabello/medical-imaging-framework/medical-imaging-framework/data/Sample cm/test_df_cm"
    vals = pd.read_csv(path + '.csv')
    
    y_val = vals['y_val']
    pred_val = vals['pred_val']

    return y_val, pred_val

def get_config(args):
    """Gets configuration information from the 'roc_config.json' file. 
    """

    with open(args.load_json) as config_file:
        roc_config = json.load(config_file)

    #Returns configurations as a dictionary
    return roc_config

def main():

    #Obtaining dictionary of configurations from json file
    parser = argparse.ArgumentParser()
    parser.add_argument('load_json', help='Load settings from file in json format.')
    args = parser.parse_args()
    roc_config = get_config(args)

    file_path = roc_config['input_path']
    results_path = roc_config['output_path']

    file_name = 'test_df_cm'

    #Obtaining needed labels and predictions
    y_val, pred_val = get_data(os.path.join(file_path, file_name))

    create_roc_curve(y_val, pred_val, roc_config, file_name, results_path)

    print("ROC curve created.")

if __name__ == "__main__":
    main()