from results_processing.roc_curve.roc_curve_many import find_directories
from results_processing.roc_curve.roc_curve import get_data
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from util.get_config import parse_json
from matplotlib import pyplot as plt
from termcolor import colored
from util import path_getter
from itertools import cycle
from scipy import interp
import pandas as pd
import numpy as np
import os


def run_program(config):
    """ Run the program for each item.

        config (dict): Program configuration.
    """
    # Get the needed input paths, as well as the proper file names for output
    pred_paths, true_paths = find_directories(config["data_path"], config['is_outer'])

    # Get the truth and prediction data for each model
    data = {}
    classes = None
    for model in pred_paths:
        data[model] = {'pred': [], 'true': []}
        for subject in pred_paths[model]:
            for item in range(len(pred_paths[model][subject])):
                true, pred = get_data(pred_paths[model][subject][item], true_paths[model][subject][item])
                data[model]['pred'] += [pred]
                data[model]['true'] += [true]
        
        # Combine the data
        data[model]['pred'] = np.concatenate(data[model]['pred'])
        data[model]['true'] = np.concatenate(data[model]['true'])
        
    # Create the ROC curves
    create_roc_curves(config, data)


def create_roc_curves(config, data):
    """ Creates ROC mean curves based on the configuration and pred/truth values.

    Args:
        config (dict): Program configuration.
        data (dict): Truth and prediction data.
    """
    # Print diagrams for each model
    for model in data:
                
        # Class labels
        classes = np.unique(data[model]['true'])
        n_classes = len(classes)
        
        # Get the truth and preds
        pred_vals = data[model]['pred']
        true_vals = label_binarize(data[model]['true'], classes=classes)
        
        # Store the true pos, false pos, and roc for both micro and macro averages
        fpr, tpr, roc_auc = ({'micro': None, 'macro': {}} for _ in range(3))
        
        # Micro Average:
        #   The precision value of all classes, as the sum of TP divided by sum of TP and FP. 
        fpr["micro"], tpr["micro"], roc_auc["micro"] = get_auc(true_vals, pred_vals)
        
        # Macro Average:
        #   Mean of the precision values for every class.
        mean_tpr = 0
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        for c in range(n_classes):
            fpr["macro"][c], tpr["macro"][c], roc_auc["macro"][c] = get_auc(true_vals[:, c], pred_vals[:, c])
            mean_tpr += np.interp(fpr_grid, fpr["macro"][c], tpr["macro"][c])
        mean_tpr /= n_classes
        fpr["macro"], tpr["macro"] = fpr_grid, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Add the micro curve to the plot
        plt.plot(
            fpr["micro"], 
            tpr["micro"],
            label='micro-average (area = {0:0.2f})' ''.format(roc_auc["micro"]),
            color=config['mean_colors']['micro'], 
            linewidth=config['line_width']
        )
        
        # Add the macro curve to the plot
        plt.plot(
            fpr["macro"], 
            tpr["macro"],
            label='macro-average (area = {0:0.2f})' ''.format(roc_auc["macro"]),
            color=config['mean_colors']['macro'], 
            linewidth=config['line_width']
        )
        
        # Plot the linear reference line
        plt.plot([0, 1], [0, 1], 'k--', lw=config['line_width'])

        # Set the diagram's font options
        plt.rcParams['font.family'] = config['font_family']
        plt.rc('axes', labelsize=config['label_font_size'])
        plt.rc('axes', titlesize=config['title_font_size'])

        # Set the diagrams's axis options
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        # Add the diagram labels
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title(f'ROC: {model} Micro and Macro Avg')
        plt.legend(loc="best")
              

        # Save the figure
        if not os.path.exists(config['output_path']):
            os.makedirs(config['output_path'])
        plt.savefig(os.path.join(config['output_path'], model) + '_roc_curve_micro_mean.' + config['save_format'], dpi=config['save_resolution'])
        plt.close()
        print(colored(f"Finished the ROC means diagram for {model}.", 'green'))
        
        
def get_auc(true_vals, pred_vals):
    """ Calculates the true pos, false pos, and area under the roc curve from true and predicted values. """
    fpr, tpr, _ = roc_curve(true_vals.ravel(), pred_vals.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json('./results_processing/roc_curve/roc_curve_many_means_config.json')
    run_program(config)


if __name__ == "__main__":
    """ Executes Program. """
    main()
