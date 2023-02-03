import sklearn
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from os import listdir
from os.path import isdir, isfile, join
from multiprocessing.pool import ThreadPool as Pool
import tensorflow as tf
import random
import pandas as pd

def get_model(width=185, height=210, depth=185):
    tf.compat.v1.reset_default_graph()
    
    """Build a 3D convolutional neural network model."""

    inputs = tf.keras.Input((width, height, depth, 1))

    x = tf.keras.layers.Conv3D(filters=20, kernel_size=10, activation="relu")(inputs)
    x = tf.keras.layers.AveragePooling3D(pool_size=5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv3D(filters=20, kernel_size=10, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)    
    x = tf.keras.layers.Conv3D(filters=50, kernel_size=10, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=10, activation="relu")(x)
    #x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(units=3, activation="softmax")(x)

    # Define the model.
    model = tf.keras.Model(inputs, outputs, name="3dcnn")
    return model