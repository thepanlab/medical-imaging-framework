import json
import random

import numpy as np
import os, sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from time import perf_counter
from tensorflow import keras
import re
import glob
import pandas as pd
import argparse
import csv

model_list = ["resnet_50", "resnet_VGG16", "Xception", "ResNet50V2", "InceptionV3"]
def get_model(model_type, target_height, target_width, channels):
    if model_type not in model_list:
        print("\033Model type not found, initializing resnet 50 insteaded.\033[0m")
    #default model restnet_50
    base_model_empty = keras.applications.resnet50.ResNet50(include_top=False,
							    weights=None,
							    input_tensor=None,
							    input_shape=(target_height, target_width, channels),
							    pooling=None)
    type = "resnet_50"

    if model_type == "resnet_VGG16":
        base_model_empty = keras.applications.VGG16(include_top=False,
					       weights=None,
					       input_tensor=None,
					       input_shape=(target_height, target_width, channels),
					       pooling=None)
        type = "resnet_VGG16"
    elif model_type == "Xception":
        base_model_empty = keras.applications.Xception(include_top=False,
						    weights=None,
						    input_tensor=None,
						    input_shape=(target_height, target_width, channels),
						    pooling=None)
        type = "Xception"
    elif model_type == "ResNet50V2":
        base_model_empty = keras.applications.ResNet50V2(include_top=False,
						weights=None,
						input_tensor=None,
						input_shape=(target_height, target_width, channels),
						pooling=None)
        type = "ResNet50V2"
    elif model_type == "InceptionV3":
        base_model_empty = keras.applications.InceptionV3(include_top=False,
							 weights=None,
							 input_tensor=None,
							 input_shape=(target_height, target_width, channels),
							 pooling=None)
        type = "InceptionV3"




    return base_model_empty, type