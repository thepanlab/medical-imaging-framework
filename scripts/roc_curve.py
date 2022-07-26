from sklearn.metrics import roc_curve, auc
import numpy as np
import os 

def roc_curve(true_vals, pred_vals, results_path, name):
    pass

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