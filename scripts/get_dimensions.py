"""get_dimensions.py

This scripts allows to determine the dimensions of images. If there is a subset with different dimensions.
It allows to get the list of files into a csv.
"""

import numpy as np
import argparse
import json
import os
import re
import pandas as pd
from PIL import Image, ImageOps
from joblib import Parallel, delayed
import sys
import multiprocessing
from collections import defaultdict
from pathlib import Path

def get_parser():
    """
    Obtains arguments parser

    Arguments:
        None
    Returns:
        ArgumentParset args
    """
    # https://www.loekvandenouweland.com/content/using-json-config-files-in-python.html
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_json', required = True,
            help='Load settings from file in json format.')

    args = parser.parse_args()

    return args
# print(parser)

def get_json(args):
    """
    Obtains configurations stores in json file

    Arguments:
        ArgumentParser: args
    Returns:
        dict config_json: it contains all the parameters from json file
    """
    with open(args.load_json) as config_file:
        config_json = json.load(config_file)

    return config_json

def get_listfiles(config_json):
    """
    It gets an array with the paths to all files

    Arguments:
        config_json (dictionary): dictionary with the parameters from json file 
    Returns:
        array of str: It contains the path to all files
    """
    file_directory = config_json["files_directory"]

    filelist = []

    for root, dirs, files in os.walk(file_directory):
        for file in files:
            #append the file name to the list
            filelist.append(os.path.join(root,file))

    a_filelist = np.array(filelist)

    return a_filelist

def get_filename(file_path):
    """
    It takes the path and split according to '/', then it takes the last part
    Arguments:
        file_address (str):  
    Returns:
        str: filename extracted from path
    """
    file_name = file_path.split("/")[-1]
    return  file_name
   
def get_n_parallel_processes(config_json):
    """
    Obtains number of cores to be used

    Arguments:
        config_json (dict): 
    Returns:
        int: number of cores that will be used 
    Raise:
        ValueError: if number of cores provided is larger than maximum
                    if number of cores is less than zero
    """
    num_cores = multiprocessing.cpu_count()

    n_parallel_processes = config_json["n_parallel_processes"]

    if n_parallel_processes == "all":
        return num_cores
    elif n_parallel_processes > num_cores:
        raise ValueError(f"Number of parallel process greater than the actual number of cores :{num_cores}")
    elif n_parallel_processes < 0:
        raise ValueError("Number of parallel process lower than zero")
    else:
        return config_json.n_parallel_processes

def get_dimension_and_filename(file_path):
    """
    Obtains the dimension and filename

    Arguments:
        file_path (str): 
    Returns:
        tuple(size 2): dimensions: height and width e.g. (1050,650)
        str: filename
    Raise:
        ValueError: if number of cores provided is larger than maximum
                    if number of cores is less than zero
    """
    img = Image.open(file_path).convert('L')       
    a_img = np.array(img)
    
    filename = get_filename(file_path)
    
    return a_img.shape, filename

def get_dimension_and_filename_all(a_filelist, config_json):
    """
    It returns list with dimensions and filename for all elements in a_filelist
    It parallelizes the calculations among different process.

    Arguments:
        a_filelist (array of str): array of path of the files
        config_json (dict): 
    Returns:
        list of tuple: list size equal to number of images
                       tuple contains two elements
                        * tuple: shape (dimensions)
                        * filename
        e.g. [((1050, 650), '9502_AIJV450-right_calyx.png'), ((1050, 650), '1441_AIJV450-right_calyx.png'), ((1050, 650), '6663_AIJV450-right_calyx.png'), ((1050, 650), '4375_AIJV450-right_calyx.png'), ((1050, 650), '682_AIJV450-right_calyx.png')]
    """

    num_cores = get_n_parallel_processes(config_json)

    results_dim_filename = Parallel(n_jobs=num_cores)( delayed(get_dimension_and_filename)(file_path) for file_path in a_filelist)

    print("Hello")

    return results_dim_filename
    
def get_qty_per_shape(results_dim_filename):
    """
    It returns list with dimensions and filename for all elements in a_filelist
    It parallelizes the calculations among different process.

    Arguments:
        a_filelist (array of str): array of path of the files
        config_json (dict): 
    Returns:
        list of tuple: list size equal to number of images
                       tuple contains two elements
                        * tuple: shape (dimensions)
                        * filename
        e.g. [((1050, 650), '9502_AIJV450-right_calyx.png'), ((1050, 650), '1441_AIJV450-right_calyx.png'), ((1050, 650), '6663_AIJV450-right_calyx.png'), ((1050, 650), '4375_AIJV450-right_calyx.png'), ((1050, 650), '682_AIJV450-right_calyx.png')]
    """
    dict_shapes = defaultdict(int)
    for shape, filename in results_dim_filename:
        # print(shape)
        dict_shapes[shape] +=1

    return dict_shapes

def get_csv_for_shape(results_dim_filename, shape_to_search, config_json):
    """
    It uses the results from `get_qty_per_shape`. It determines the filenames that have
    a specific shape and store them in a csv file with name `df_list_shape_{shape_to_search}.csv`

    Arguments:
        results_dim_filename (list of tuple): lists with dimensions and filename
        shape_to_search (tuple len 2): dimension: height and width. e.g. (1050, 650)
        config_json (dict): 
    Returns:
    """
    l_shape = []
    for shape, filename in results_dim_filename:
    # print(shape)
        if shape == shape_to_search:
            l_shape.append(filename)
    
    df = pd.DataFrame(l_shape, columns= ["filename"])

    output_directory =  config_json["output_results"]
    
    df.to_csv(Path(output_directory,f"df_list_shape_{shape_to_search}.csv"))
            
def main():
    args = get_parser()
    config_json = get_json(args)
    a_filelist = get_listfiles(config_json)
    l_dim_and_filename = get_dimension_and_filename_all(a_filelist, config_json)
    print(get_qty_per_shape(l_dim_and_filename))
    get_csv_for_shape(l_dim_and_filename, (421, 650), config_json)

if __name__ == "__main__":
    main()