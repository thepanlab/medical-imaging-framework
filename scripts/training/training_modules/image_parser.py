from abc import ABC, abstractclassmethod, abstractmethod
from tensorflow import keras as K
import tensorflow as tf
from skimage import io
import numpy as np


def parse_image(filename, mean, use_mean, class_names, channels, do_cropping, offset_height, offset_width, target_height, target_width, label_position=None, use_labels=True): 
    """ Parses an image from some given filename and various parameters.
        
        -- Input Parameters ------------------------
        filename (Tensor str): A tensor of some file name.
        mean (double): A mean value.
        
        use_mean (bool): Whether to use the mean to normalize.
        class_names (list of str): A list of label class names.
        
        channels (int): Channels in which to decode image. 
        do_cropping (bool): Whether to crop the image.
        offset_height (int): Image height offset.
        
        offset_width (int): Image width offset.
        target_height (int): Image height target.
        target_width (int): Image width target.
        
        label_position (int): Optional. The position of the label in the image name. Default is None.
        use_labels (bool): Optional. Whether to consider/output the true image label. Default is True.
        --------------------------------------------
        
        -- Return ----------------------------------
        (Tensor image): An image.
        (Tensor str): The true label of the image.
        --------------------------------------------
    """
    # Assert eager execution, else trouble will be had...
    if not tf.executing_eagerly():
        raise Exception(
            "Fatal Error: TensorFlow must have eager execution enabled when parsing images.\n\t" +
            "Try including this function call within 'tf.py_function' and/or calling\n\t" +
            "'tf.config.run_functions_eagerly(True)' before running this."
        )
    
    # Split to get only the image name
    image_path = tf.strings.split(filename, "/")[-1]
    
    # Remove the file extention
    path_substring = tf.strings.regex_replace(
        image_path,
        ".png|.jpg|.jpeg|.tiff|.csv", 
        ""
    )
    
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels=channels, dtype=tf.float32, name=None, expand_animations=False)
    
    # TODO: Perhaps make image parser one function, so it can be changed in one area?
    # image_type = tf.strings.split(filename, ".")[-1].numpy().decode()
    # if image_type == 'csv':
        # TODO: do this
    # else:
        # TODO: do that
    
    # Crop the image
    if do_cropping:
        image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
       
    # Normalize the image
    if use_mean:
        image = image - mean / 255
    
    # Find the label if needed
    if use_labels:
        path_label = tf.strings.split(path_substring, "_")[label_position]
        class_name = tf.argmax(path_label == class_names)
        return image, class_name
    else:
        return image
    

class ImageReader(ABC):
    """ Abstract ImageReader Class """
    def __init__(self):
        return
    
    @abstractmethod
    def io_read(self, filename):
        """ Reads an image from a file and loads it into memory """
        pass

    @abstractmethod
    def parse_image(self, filename, mean, use_mean, class_names, channels, do_cropping, offset_height, offset_width, target_height, target_width, label_position=None, use_labels=True): 
        pass
    
    
class ImageReaderGlobal(ImageReader):
    """ Handles reading and loading of files that can be read using io.imread
            Includes:
                - .jpg
                - .jpeg
                - .png
                - .tiff
    """
    def __init__(self):
        ImageReader.__init__(self)
        return

    def io_read(self, filename):
        im=io.imread(filename.numpy().decode())
        im=im.reshape(185,210,185,1)
        return im

    def parse_image(self, filename, mean, use_mean, class_names, channels, do_cropping, offset_height, offset_width, target_height, target_width, label_position=None, use_labels=True): 
        # Split to get only the image name
        image_path = tf.strings.split(filename, "/")[-1]
        
        # Remove the file extention
        path_substring = tf.strings.regex_replace(
            image_path,
            ".png|.jpg|.jpeg|.tiff|.csv", 
            ""
        )
        
        # Find the label
        label = tf.strings.split(path_substring, "_")[label_position]
        label_bool = (label == class_names)

        # Read in the image
        image = self.io_read(filename)
        
        # Crop the image
        if do_cropping == 'true':
            image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
        
        # Normalize the image
        if use_mean == 'true':
            image = image - mean / 255

        return image, tf.argmax(label_bool)
    
    
class ImageReaderCSV(ImageReader):
    """ Handles reading and parsing of .csv images

        CSV images are special as they cannot be read by io.imread()
        Stored as 1-Dimensional arrays, when they are loaded, they will need to be reshaped
    """
    def __init__(self, configs):
        self._configs = configs
        ImageReader.__init__(self)
        return

    def io_read(self, filename):
        image = np.genfromtxt(filename)
        if "csv_shape" in self._configs:
            image = image.reshape(self._configs["csv_shape"])
        else:
            image = image.reshape(185, 210, 185, 1)

        return image

    def parse_image(self, filename, mean, use_mean, class_names, channels, do_cropping, offset_height, offset_width, target_height, target_width, label_position=None, use_labels=True): 
        # Split to get only the image name
        image_path = tf.strings.split(filename, "/")[-1]
        
        # Remove the file extention
        path_substring = tf.strings.regex_replace(
            image_path,
            ".png|.jpg|.jpeg|.tiff|.csv", 
            ""
        )
        
        # Find the label
        label = tf.strings.split(path_substring, "_")[label_position]
        label_bool = (label == class_names)

        image = self.io_read(filename)

        # Crop the image
        if do_cropping == 'true':
            image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
        
        # Normalize the image
        if use_mean == 'true':
            image = image - mean / 255
        
        return image, tf.argmax(label_bool)
    