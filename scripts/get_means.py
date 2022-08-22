#!/usr/bin/env python
# coding: utf-8

# import sys
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

# Testing first file to see if it contained the specific letter for subject

def get_filename(file_address):
    """
    It takes the path and split according to '/', then it takes the last part
    Arguments:
        file_address (str):  
    Returns:
        str: filename extracted from path
    """
    file_name = file_address.split("/")[-1]
    return  file_name

def get_subject(file_address, letter_search):
    """
    It gets the subject from the file address,
    in order to recognize it uses the letter that identify the subject.
    It searches for the combination of letter with number

    Arguments:
        file_address (str):  
        letter_search (str): 
    Returns:
        str: it returns letter_search plus numbers found
    Raise:
        ValueError: If the letter provided is not found preceding a number
    """
    x = re.search(letter_search + "[0-9]+", get_filename(file_address) )

    if x is None:
        raise ValueError(f"Combination of {letter_search} with number was not found in file address {file_address}")

    return x.group(0)

def get_array_subjects_and_unique(a_filelist, config_json):
    """
    It gets array of subjects for each images as well as a list of uniques subjects
    contained in all images

    Arguments:
        a_filelist (array of str):  
        config_json (dict): 
    Returns:
        array of str: value of subject for each image
        list of str: list with unique subjects
    """

    letter = config_json["letter"]

    l_subjects = []

    for item in a_filelist:
        l_subjects.append(get_subject(item, letter))        

    a_subjects = np.array(l_subjects)

    set_subject = set(l_subjects)
    l_unique_subject = list(set_subject)

    # TODO
    # Fix order by number after k
    # e.g.
    # Currenlty the sort list gives
    # k1,k10,k2,k3, ....
    l_unique_subject.sort()

    return a_subjects, l_unique_subject


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


def cv_inner_loop(index_test, index_val, a_filelist, a_subjects, l_unique_subject):
    """
    It selects the subset of images for the inner loop
    for a specific index_test and index_val
    Then, it calculates the mean of the grayscale images

    Arguments:
        index_test (str): test subject
        index_val (str): validation subject
        a_filelist (array of str): array of path of the files
        a_subjects (array of str): array of the subject for each image
        l_unique_subject (list of str): list of unique subjects
        
    Returns:
        float: the mean 
    """

    bool_sub_train_val = a_subjects != index_test

    a_filelist_train_val = a_filelist[bool_sub_train_val]
    a_subject_train_val = a_subjects[bool_sub_train_val]

    a_num_val = np.array(l_unique_subject)
    a_num_test = np.array([index_test])
    
    a_num_val = np.setdiff1d(a_num_val,a_num_test)

    print("Test", a_num_test)
    print("Val", a_num_val)

    print(f"Epidural_val: {index_val}")
    bool_val = ( a_subject_train_val == index_val )
    bool_train = ~bool_val
       
    a_filelist_train, a_filelist_val = a_filelist_train_val[bool_train], a_filelist_train_val[bool_val]

    mean_sum = 0
    count = 1
    
    for img_address in a_filelist_train:
        img_temp = Image.open(img_address).convert('L')
        
        img_temp_array = np.array(img_temp)

        mean_temp = img_temp_array.mean()

        mean_sum += mean_temp

        if count %100 == 0:
            print(count,"/",len(a_filelist_train), end='\r')

        count+= 1
    print()
        
    mean_temp = mean_sum/len(a_filelist_train)

    return mean_temp

def get_inner_loop_combinations(l_unique_subject):
    """
    It creates all the possible combinations for 
    test and validation for the subjects

    Arguments:
        l_unique_subject (list of str): list of unique subjects
        
    Returns:
        list of tuples: the combinations 
    """
    list_combinations = []

    for index_test in l_unique_subject:

        a_num_val = np.array(l_unique_subject)
        a_num_test = np.array([index_test])

        a_num_val = np.setdiff1d(a_num_val,a_num_test)

        list_combinations.extend(list(zip([index_test for i in range(len(a_num_val))], a_num_val)))
    
    return  list_combinations

def get_inner_loop_means_csv(l_unique_subject, a_filelist, a_subjects, config_json):
    """
    It creates a csv file for the inner loop results
    int the directory specified in the json file

    The file name is "means_nested_cv_inner_loop.csv"

    Arguments:
        l_unique_subject (list of str): list of unique subjects
        a_filelist (array of str): array of path of the files
        a_subjects (array of str): array of the subject for each image
        config_json (dict): 

    Returns:
        None
    """
    output_directory = config_json["output_means"]
    num_cores = get_n_parallel_processes(config_json)

    inner_combinations = get_inner_loop_combinations(l_unique_subject)

    results_mean = Parallel(n_jobs=num_cores)( delayed(cv_inner_loop)(index_test, index_val, a_filelist, a_subjects,l_unique_subject) \
                                        for index_test, index_val in inner_combinations )

    columns = [f"V_{ite_subject}" for ite_subject in l_unique_subject]
    indices = [f"T_{ite_subject}" for ite_subject in l_unique_subject]

    df_mean_inner = pd.DataFrame(columns = columns, 
                        index = indices)

    for i_combinations, [index_test, index_val] in enumerate(inner_combinations):
        df_mean_inner.loc[f"T_{index_test}", f"V_{index_val}"] = results_mean[i_combinations]
    
    # This allow to deal with directories with and without /
    # os.path.join(output_directory,"means_nested_cv_inner_loop.csv")
    # Create subdirectory if it doesnt exit
    os.makedirs(output_directory, exist_ok=True)

    df_mean_inner.to_csv( os.path.join(output_directory,"means_nested_cv_inner_loop.csv") )

    return

def get_outer_loop_means_csv(l_unique_subject, a_filelist, a_subjects, config_json):
    """
    It creates a csv file for the outer loop results
    int the directory specified in the json file

    The file name is "means_nested_cv_outer_loop.csv"

    Arguments:
        l_unique_subject (list of str): list of unique subjects
        a_filelist (array of str): array of path of the files
        a_subjects (array of str): array of the subject for each image
        config_json (dict): 

    Returns:
        None
    """
    output_directory =  config_json["output_means"]
    num_cores = get_n_parallel_processes(config_json)

    columns = ["mean_test"]
    indices = [f"T_{ite_subject}" for ite_subject in l_unique_subject]

    df_mean_outer = pd.DataFrame(columns = columns, 
                    index = indices)

    results_mean = Parallel(n_jobs=num_cores)( delayed(cv_outer_loop)(index_test, a_filelist, a_subjects) \
                                        for index_test in l_unique_subject )

    for i_subject, index_test in enumerate(l_unique_subject):
        df_mean_outer.loc[f"T_{index_test}","mean_test"] = results_mean[i_subject]

    df_mean_outer.to_csv( os.path.join(output_directory,"means_nested_cv_outer_loop.csv") )


def cv_outer_loop(index_test, a_filelist, a_subjects):
    """
    It selects the subset of images for the outer loop
    for a specific index_test
    Then, it calculates the mean of the grayscale images

    Arguments:
        index_test (str): test subject
        a_filelist (array of str): array of path of the files
        a_subjects (array of str): array of the subject for each image
        l_unique_subject (list of str): list of unique subjects
        
    Returns:
        float: the mean 
    """

    bool_sub_train_val = a_subjects != index_test

    a_filelist_train_val = a_filelist[bool_sub_train_val]
    
    print("Test", index_test)

    mean_sum = 0
    count = 1

    for img_address in a_filelist_train_val:
        img_temp = Image.open(img_address).convert('L')

        img_temp_array = np.array(img_temp)

        mean_temp = img_temp_array.mean()

        mean_sum += mean_temp

        if count %100 == 0:
            print(count,"/",len(a_filelist_train_val), end='\r')

        count+= 1

    print()        

    mean_temp = mean_sum/len(a_filelist_train_val)

    return mean_temp


def main():
    args = get_parser()
    config_json = get_json(args)
    a_filelist = get_listfiles(config_json)
    a_subjects, l_unique_subject =  get_array_subjects_and_unique(a_filelist, config_json)
    get_inner_loop_means_csv(l_unique_subject, a_filelist, a_subjects, config_json)
    get_outer_loop_means_csv(l_unique_subject, a_filelist, a_subjects, config_json)

if __name__ == "__main__":
    main()