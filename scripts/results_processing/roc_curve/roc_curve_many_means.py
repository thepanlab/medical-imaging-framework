from results_processing.roc_curve.roc_curve_many import find_directories
from results_processing.roc_curve.roc_curve import get_data
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
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
    
    # Read in the data from the filepaths
    data, classes = read_data(config, pred_paths, true_paths)
        
    # Create the ROC curves
    create_roc_curves(config, data, classes)
    
    
def read_data(config, pred_paths, true_paths):
    """ Reads the data from the input files """
    # For every model...
    data = {}
    classes = []
    for model in pred_paths:
        data[model] = {}
        
        # Read in the data for each subject...
        for subject in pred_paths[model]:
            if config['average_all_subjects'] or subject in config['subjects']:
                data[model][subject] = {'pred': [], 'true': []}
                
                # Process the data from each file path.
                for item in range(len(pred_paths[model][subject])):
                    true, pred = get_data(pred_paths[model][subject][item], true_paths[model][subject][item])
                    data[model][subject]['pred'] += [pred]
                    data[model][subject]['true'] += [true]
                    cat = np.concatenate(data[model][subject]['true'])
                    classes += [c for c in np.unique(cat) if c not in classes]
    return data, classes
        
        
def get_auc(true, pred):
    """ Calculates the true pos, false pos, and area under the roc curve from true and predicted values. """
    fpr, tpr, _ = roc_curve(true.ravel(), pred.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def get_micro_average(data, classes):
    """ For every class, calculate the rates """
    # Combine all of the subjects into a single array
    true_cat, pred_cat = [], []
    for subject in data:
        true_cat += data[subject]['true']
        pred_cat += data[subject]['pred']
    true_cat, pred_cat = label_binarize(np.concatenate(true_cat), classes=classes), np.concatenate(pred_cat)
    
    # Get the items for each class
    fpr, tpr, roc_auc = ({} for _ in range(3))
    for i in range(len(classes)):
        fpr[i], tpr[i], roc_auc[i] = get_auc(true_cat[:, i], pred_cat[:, i])
    return fpr, tpr, roc_auc


def get_macro_average(data, classes):
    """ For every subject calculate the rates, then mean the resulting values """
    # Think of the subjects as the 'classes' when taking the F/TPR averages.
    #   Get the mean of each subject-label, then mean on all of the subjects per label.
    subjects = [s for s in data]
    n_subjects = len(subjects)
    
    # For each class...
    fpr, tpr, roc_auc = ({} for _ in range(3))
    for i in range(len(classes)):
        fpr[i], tpr[i], roc_auc[i] = ({} for _ in range(3))
        
        # Find the items for each individual subject.
        for s in range(n_subjects):
            true = label_binarize(np.concatenate(data[subjects[s]]['true']), classes=classes)[:, i]
            pred = np.concatenate(data[subjects[s]]['pred'])[:, i]
            fpr[i][s], tpr[i][s], roc_auc[i][s] = get_auc(true, pred)
            
        # Calculate the mean of the subjects in each class
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for s in range(n_subjects):
            mean_tpr += np.interp(fpr_grid, fpr[i][s], tpr[i][s])
        mean_tpr /= n_subjects
            
        # Get the items
        fpr[i] = fpr_grid
        tpr[i] = mean_tpr
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc
        

def create_roc_curves(config, data, classes):
    """ Creates ROC mean curves based on the configuration and pred/truth values.

    Args:
        config (dict): Program configuration.
        data (dict): Truth and prediction data.
    """    
    # Print diagrams for each model
    for model in data:
                
        # Calculate the averages, seperated by class
        fpr_micro, tpr_micro, roc_auc_micro = get_micro_average(data[model], classes)
        fpr_macro, tpr_macro, roc_auc_macro = get_macro_average(data[model], classes)
        
        # Plot a graph for each mean-type
        for plot_i in ['micro', 'macro']:
            
            # Plot the micro curve values
            if plot_i == 'micro':
                for i, type_name, color in zip(classes, config['label_types'], config['line_colors']):
                    plt.plot(
                        fpr_micro[i], 
                        tpr_micro[i],
                        label='{0} (AUC={1:0.2f})' ''.format(type_name, roc_auc_micro[i]),
                        color=color, 
                        linewidth=config['line_width']
                    )
            
            # Plot the macro curve values
            else:
                for i, type_name, color in zip(classes, config['label_types'], config['line_colors']):
                    plt.plot(
                        fpr_macro[i], 
                        tpr_macro[i],
                        label='{0} (AUC={1:0.2f})' ''.format(type_name, roc_auc_macro[i]),
                        color=color, 
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
            plt.title(f'ROC: {model} {plot_i} average')
            plt.legend(loc="best")
                
            # Save the figure
            if not os.path.exists(config['output_path']):
                os.makedirs(config['output_path'])
            plt.savefig(f'{os.path.join(config["output_path"], model)}_roc_curve_{plot_i}_mean.{config["save_format"]}', dpi=config['save_resolution'])
            plt.close()
            print(colored(f"Finished the ROC {plot_i} means diagram for {model}.", 'green'))


def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json('./results_processing/roc_curve/roc_curve_many_means_config.json')
    run_program(config)


if __name__ == "__main__":
    """ Executes Program. """
    main()
