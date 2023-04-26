from IPython.display import Image
from termcolor import colored
from tensorflow import keras
from util import get_config
import matplotlib.cm as cm
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import os


def get_images(addr):
    """ Get the image address(es)

    Args:
        addr (str): A path to an image or a folder.

    Returns:
        list: A list of images.
    """
    # If a single image
    if addr.endswith(".jpg") or addr.endswith(".png"):
        return [addr]

    # If a folder
    images = []
    for file in os.listdir(addr):
        name = os.fsdecode(file)
        if name.endswith(".jpg") or name.endswith(".png"):
            images.append(os.path.join(addr, name))
    print(colored("There are " + str(len(images)) + " image(s) to be processed.", 'green'))
    return images


def load_data(model_addr):
    """ Load the data from their addresses

    Args:
        model_addr (str): Path to the model.

    Returns:
        tuple: The loaded model.
    """
    model = keras.models.load_model(model_addr)
    model.layers[-1].activation = None
    return model


def preprocessing(img_addr):
    """ Preprocess the image

    Args:
        img_addr (str): The image address.

    Returns:
        image: An altered image.
    """
    img = Image.open(img_addr).convert('L') #.resize((181, 241))
    return np.array(img)[np.newaxis, :, :, np.newaxis]

def check_layer_name(layer_names, name):
    """ Gets the name of a layer, if given the correct input.

    Args:
        layer_names (list of str): A list of layer names.
        name (str): The name of the target

    Returns:
        str: Layer name if valid, else an empty string.
    """
    try:
        if name:
            index_global = layer_names.index(name)
            return layer_names[index_global-1]
    except:
        print(colored("Warning: the given layer name is invalid.", 'yellow'))
        return ""


def get_layer_name(model, last_conv_layer_name):
    """ Gets the layer name of the target.

    Args:
        model (keras.Model): The input model to derive the layers from.
        last_conv_layer_name (str): The target layer.
    """
    layer_names = [layer["config"]["name"] for layer in model.get_config()["layers"]]
    layer_names.sort()
    
    last_conv_layer_name = check_layer_name(layer_names, last_conv_layer_name)
    if not last_conv_layer_name:
        print(model.summary())
        for name in layer_names:
            print(colored(f"\t{name}", 'cyan'))
        print(colored("Please choose a layer name from the above list:", 'magenta'))
        while not last_conv_layer_name:
            last_conv_layer_name = input()
            last_conv_layer_name = check_layer_name(layer_names, last_conv_layer_name)
    return last_conv_layer_name


def gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None):
    """ Generate a Grad-CAM gradcam

    Args:
        img (image): An image to create a gradcam from.
        model (keras.models): A trained model.
        last_conv_layer_name(str): The convolutional layer name.
        pred_index (int, optional): Selects the class/category to create the visualization . Default None mean predicted class.

    Returns:
        image: A heatmap image.
    
    """
    
    # Get the last convolutional layer name
    last_conv_layer_name = get_layer_name(model, last_conv_layer_name)
    
    # First, we create a model that maps the input image to the activations
    #   of the last conv layer as well as the output predictions.
    #  Input 0 of layer expected shape=(None, 241, 181, 1)
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
    """ Save the Grad-CAM output to a file.

    Args:
        img_path (str): Path to save image to.
        heatmap (image): The processed image.
        cam_path (str): The output directory.
        alpha (float, optional): The superimposed image alpha. Defaults to 0.4.
    """
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
    superimposed_img.save(os.path.join(cam_path, new_img_name))


def main(config=None):
    """ The main program.

    Args:
        config (dict, optional): A custom configuration. Defaults to None.
    """
    # Obtaining dictionary of configurations from the json file
    if config is None:
        config = get_config.parse_json('./results_processing/grad_cam/grad_cam_config.json')

    # Load the data needed for the program
    model = load_data(config["input_model_address"])

    # If a folder of images, read the folder or else just use the one image
    img_addrs = get_images(config["input_img_address"])

    # Process every image
    for img_addr_i in img_addrs:
        # Preprocess the image
        img_processed = preprocessing(img_addr_i)

        # Generate a heatmap from the model
        heatmap = gradcam_heatmap(img_processed, model, config['last_conv_layer_name'], None if config["index_class_gradcam"] == -1 else config["index_class_gradcam"])

        # Output the image result
        save_gradcam_output(img_addr_i, heatmap, cam_path=config["output_image_address"], alpha=config["alpha"])
        print(colored("Finished processing: " + img_addr_i, 'green'))


if __name__ == "__main__":
    """ Executes the program """
    main()
