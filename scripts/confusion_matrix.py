from sklearn.metrics import confusion_matrix
import numpy as np
import os 

def create_confusion_matrix(true_vals, pred_vals, results_path, name):

    conf_matrix = confusion_matrix(true_vals, pred_vals)

    np.savetxt(results_path + name + '_conf_matrix', conf_matrix)

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

    create_confusion_matrix()

if __name__ == "__main__":
    main()