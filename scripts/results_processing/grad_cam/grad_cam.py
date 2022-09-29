from IPython.display import Image
from termcolor import colored
from tensorflow import keras
from util import get_config
import matplotlib.cm as cm
from os.path import exists
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import os

"""
    Grad-CAM
    
    This file takes in a JSON configuration file that lists:
        - input_model_address: The file location of a trained model.
        - input_means_address: A csv file of mean values (optional)
        - input_img_address: A folder or specific image file to read
        - output_image_address: The file location to place output heatmaps into.
        - alpha: Changes the output image's appearance.
        - mean_row: The row-location of the mean value in the csv.
        - mean_col: The column-location of the mean value in the csv.
        - last_conv_layer_name: The last layer of activations in the model.
        
    It will display a heatmap of the most influential points on an image according
    to the model. Each converted image is saved to a file of a similar name.
"""


def load_config():
    """ Load the JSON configuration file """
    # Get the configuration
    config = get_config.parse_json('grad_cam_config.json')

    # Check if the given input files exist
    files = ['input_model_address', 'input_means_address', 'input_img_address']
    for file in files:
        addr = config[file]
        if not exists(addr):
            raise Exception(colored("Error: The following file does not exist! " + addr))
        if os.stat(addr).st_size == 0:
            raise Exception("Error: The following file is empty! " + addr)

    # Return the json contents as variables
    return (
        config['input_model_address'],
        config['input_means_address'],
        config['input_img_address'],
        config['output_image_address'],
        config['alpha'],
        config['mean_row'],
        config['mean_col'],
        config['last_conv_layer_name']
    )


def get_images(addr):
    """ Get the image address(es) """
    # If a single image
    if addr.endswith(".jpg") or addr.endswith(".png"):
        print(colored("There is one image to be processed.", 'green'))
        return [addr]

    # If a folder
    images = []
    for file in os.listdir(addr):
        name = os.fsdecode(file)
        if name.endswith(".jpg") or name.endswith(".png"):
            images.append(os.path.join(addr, name))
    print(colored("There are " + str(len(images)) + " image(s) to be processed.", 'green'))
    return images


def load_data(model_addr, mean_addr, mean_row, mean_col):
    """ Load the data from their addresses """
    model = keras.models.load_model(model_addr)
    mean = pd.read_csv(mean_addr, index_col=0)
    mean = mean.loc[mean_row, mean_col]
    return model, mean


def preprocessing(img_addr, mean):
    """ Preprocess the image """
    img = np.array(Image.open(img_addr).convert('L'))
    if mean is not None:
        img_centered = img - mean
    else:
        img_centered = img
    return img_centered[np.newaxis, :, :, np.newaxis]


def gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None):
    """ Generate a Grad-CAM heatmap """
    # First, we create a model that maps the input image to the activations
    #   of the last conv layer as well as the output predictions.
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    #   with respect to the activations of the last conv layer.
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    #   with regard to the output feature map of the last conv layer.
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    #   over a specific feature map channel.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    #   by "how important this channel is" with regard to the top predicted class
    #   then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap


def save_gradcam_output(img_path, heatmap, cam_path, alpha=0.4):
    """ Save the Grad-CAM output to a file """
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Get the new image name
    input_img_name = img_path.split('/')[-1]
    base_img_name = input_img_name.split('.')[0]
    new_img_name = base_img_name + "_heatmap.jpg"

    # Save the superimposed image
    if not os.path.exists(cam_path):
        os.makedirs(cam_path)
    if cam_path.endswith('/'):
        superimposed_img.save(cam_path + new_img_name)
    else:
        superimposed_img.save(cam_path + '/' + new_img_name)


def main():
    """ Main program """
    # Load the configuration file and data
    model_addr, mean_addr, img_addr, output_addr, alpha, \
    mean_row, mean_col, last_conv_layer_name = load_config()

    # Load the data needed for the program
    model, mean = load_data(model_addr, mean_addr, mean_row, mean_col)

    # If a folder of images, read the folder or else just use the one image
    img_addrs = get_images(img_addr)

    # Process every image
    for img_addr_i in img_addrs:
        # Preprocess the image
        img_processed = preprocessing(img_addr_i, mean)

        # Generate a heatmap from the model
        heatmap = gradcam_heatmap(img_processed, model, last_conv_layer_name)

        # Output the image result
        save_gradcam_output(img_addr_i, heatmap, cam_path=output_addr, alpha=alpha)
        print(colored("Finished processing: " + img_addr_i, 'green'))


if __name__ == "__main__":
    """ Executes the program """
    main()
