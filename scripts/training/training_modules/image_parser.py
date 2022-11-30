import numpy as np
import tensorflow as tf

from tensorflow import keras as K
from skimage import io
from abc import ABC, abstractclassmethod


""" Abstract ImageReader class
"""
class ImageReader(ABC):
    def __init__(self):
        return
    
    @abstractmethod
    def io_read(self, filename):
        """ Reads an image from a file and loads it into memory
        """
        pass

    @abstractmethod
    def parse_image(self, filename, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width):
        """ Parses an image from some given filename and various parameters.
        
        -- Input Parameters ------------------------
        filename (Tensor str): A tensor of some file name.
        mean (double): A mean value.
        
        use_mean (bool): Wather to use the mean to normalize.
        class_names (list of str): A list of label class names.
        label_position (int): The position of the label in the image name.
        
        channels (int): Channels in which to decode image. 
        do_cropping (bool): Whether to crop the image.
        offset_height (int): Image height offset.
        
        offset_width (int): Image width offset.
        target_height (int): Image height target.
        target_width (v): Image width target.
        --------------------------------------------
        
        -- Return ----------------------------------
        (Tensor image): An image.
        (Tensor str): The true label of the image.
        --------------------------------------------
        """
        pass


""" Handles reading and loading of files that can be read using io.imread
    Includes:
        - .jpg
        - .jpeg
        - .png
        - .tiff
"""
class ImageReaderGlobal(ImageReader):
    def __init__(self):
        ImageReader.__init__(self)
        return

    def io_read(self, filename):
        return tf.py_function(io.imread, filename.numpy().decode(), tf.float32)

    def parse_image(self, filename, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width):
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


""" Handles reading and parsing of .csv images

    CSV images are special as they cannot be read by io.imread()
    Stored as 1-Dimensional arrays, when they are loaded, they will need to be reshaped
"""
class ImageReaderCSV(ImageReader):
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

    def parse_image(self, filename, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width):
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