from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from termcolor import colored
from itertools import cycle
import pandas as pd
import numpy as np
import get_config
import os


# TODO: change title to include file name


def create_roc_curve(true_vals, pred_vals, roc_config, file_name, output_path):
    """ Creates the ROC Graph. """
    # Assuming true_vals is a numpy array
    classes = np.unique(true_vals)
    n_classes = len(classes)

    # Binarize the labels
    true_val_bin = label_binarize(true_vals, classes=classes)

    # Create dictionaries for true pos, false pos, and roc values
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # For each class-index, generate values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_val_bin[:, i], pred_vals[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Generate values for a ravelled array
    fpr["micro"], tpr["micro"], _ = roc_curve(true_val_bin.ravel(), pred_vals.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
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

    # Establishing the plot's font and font sizes
    font_family = roc_config['font_family']
    label_font_size = roc_config['label_font_size']
    title_font_size = roc_config['title_font_size']

    # Setting the previously established parameters
    plt.rcParams['font.family'] = font_family
    plt.rc('axes', labelsize=label_font_size)
    plt.rc('axes', titlesize=title_font_size)

    # Removing top and right axis lines
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    # Create the plot
    for i, type_name, color in zip(classes, l_types, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (AUC = {1:0.2f})'
                       ''.format(type_name, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver Operating Characteristic(ROC): ' + file_name)
    plt.legend(loc="best")

    # Save the figure
    plt.savefig(os.path.join(output_path, file_name) + '_roc_curve.' + save_format, dpi=save_res)
    plt.close()


def get_data(pred_path, true_path):
    """ Reads in the labels and predictions from CSV. """
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


def main(config=None):
    """ The main program. """

    # Obtaining dictionary of configurations from json file
    if config is None:
        config = get_config.parse_json('roc_curve_config.json')

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

    # Create the curve
    create_roc_curve(true_val, pred_val, config, config['output_file_prefix'], output_path)
    print(colored("ROC curve created for: " + config['output_file_prefix'], "green"))


if __name__ == "__main__":
    """ Executes the program """
    main()
