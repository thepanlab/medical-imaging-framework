from sklearn.metrics import roc_curve, auc
import numpy as np
import os 

def roc_curve(true_vals, pred_vals, results_path, name):

    #assuming true_vals is a numpy array
    n_classes = np.unique(true_vals)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_vals[:, i], pred_vals[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(true_vals.ravel(), pred_vals.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

def get_data(path):

    with open(path + '.npy', 'rb') as f:
        pred_vals = np.load(f)

    return pred_vals

def main():

    #Establishing needed directories
    dir_path = os.getcwd()
    data_path = dir_path + '/data/Sample Data/'
    results_path = dir_path + '/results/'
    file_name = 'test'

    roc_curve()

if __name__ == "__main__":
    main()