#!/usr/bin/env python
# coding: utf-8

import numpy as np
import glob
import argparse
import sys
import json
import os
import re
import pandas as pd
from PIL import Image, ImageOps

# https://www.loekvandenouweland.com/content/using-json-config-files-in-python.html
parser = argparse.ArgumentParser()

parser.add_argument('load_json',
        help='Load settings from file in json format.')

l_images = glob.glob("./data/ep_intensity_all/*")
a_images = np.array(l_images)

# print(parser)

args = parser.parse_args()

# print(args)
# print(args.load_json)

with open(args.load_json) as config_file:
    data = json.load(config_file)

file_directory = data["files_directory"]
output_directory = data["output_means"]

# print(file_directory)

filelist = []

for root, dirs, files in os.walk(file_directory):
    for file in files:
        #append the file name to the list
        filelist.append(os.path.join(root,file))

# print(filelist)

a_filelist = np.array(filelist)

letter = data["letter"]

l_subjects = []

# Testing first file to see if it contained the specific letter for subject

def get_filename(file_address):
    file_name = file_address.split("/")[-1]
    return  file_name

def get_subject(file_address, letter_search):
    x = re.search(letter_search + "[0-9]+", get_filename(file_address) ) 
#     print(x.group(0))
    return x.group(0)
   
def check_letter_in_filename(file_address, letter_search):
    x = re.search(letter_search + "[0-9]+", get_filename(file_address) ) 
    
    if x == None:
        raise Exception("Letter {} not found in file name {}".format(letter, file_address)) 

# Just to check that the letter is present in the first file
check_letter_in_filename(filelist[0], letter)

l_subjects = []

for item in filelist:
    
    l_subjects.append(get_subject(item, letter))        

a_subjects = np.array(l_subjects)

set_subject = set(l_subjects)

# for i in set_subject:
#     print(i)

l_unique_subject = list(set_subject)

l_unique_subject.sort()

# ## Mean Nested CV: Inner loop

# ### Auto

n_subjects = len(l_unique_subject)

columns = ["V_" + ite_subject for ite_subject in l_unique_subject]
indices = ["T_" + ite_subject for ite_subject in l_unique_subject]

df_mean = pd.DataFrame(columns = columns, 
                       index = indices)


for i in range(len(l_unique_subject)):

    index_test = l_unique_subject[i]
    bool_sub_train_val = a_subjects != index_test

    a_filelist_train_val = a_filelist[bool_sub_train_val]
    a_subject_train_val = a_subjects[bool_sub_train_val]

    a_num_val = np.array(l_unique_subject)
    a_num_test = np.array([index_test])
    
    a_num_val = np.setdiff1d(a_num_val,a_num_test)

    print("Test", a_num_test)
    print("Val", a_num_val)
    
#     break
    
    for index_val in a_num_val:

        print("Epidural_val: " + str(index_val) )
        bool_val = ( a_subject_train_val == index_val )
        bool_train = ~bool_val
        
#         break
        
        a_filelist_train, a_filelist_val = a_filelist_train_val[bool_train], a_filelist_train_val[bool_val]
#         break     
        mean_sum = 0
        count = 1
        
        for img_address in a_filelist_train:
            img_temp = Image.open(img_address).convert('L')
            
            img_temp_array = np.array(img_temp)

            mean_temp = img_temp_array.mean()
#             print(mean_temp)
            mean_sum += mean_temp
#             break
            if count %100 == 0:
                print(count,"/",len(a_filelist_train), end='\r')

            count+= 1
        print()
            
        mean_temp = mean_sum/len(a_filelist_train)
        
        df_mean.loc["T_"+index_test,"V_"+index_val] = mean_temp 
       
# This allow to deal with directories with and without /
# os.path.join(output_directory,"means_nested_cv_inner_loop.csv")

# Create subdirectory if it doesnt exit
os.makedirs(output_directory, exist_ok=True)

df_mean.to_csv( os.path.join(output_directory,"means_nested_cv_inner_loop.csv") )