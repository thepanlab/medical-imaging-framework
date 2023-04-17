from abc import ABC, abstractmethod
from termcolor import colored
import tensorflow as tf
from skimage import io
import numpy as np


def parse_image(filename, class_names, channels, do_cropping, offset_height, offset_width, target_height, target_width, label_position=None, use_labels=True): 
    """ Parses an image from some given filename and various parameters.
        
    Args:
        filename (Tensor str): A tensor of some file name.
        class_names (list of str): A list of label class names.
        
        channels (int): Channels in which to decode image. 
        do_cropping (bool): Whether to crop the image.
        offset_height (int): Image height offset.
        
        offset_width (int): Image width offset.
        target_height (int): Image height target.
        target_width (int): Image width target.
        
        label_position (int): Optional. The position of the label in the image name. Must provide if using labels. Default is None.
        use_labels (bool): Optional. Whether to consider/output the true image label. Default is True.
        
    Returns:
        (Tensor image): An image.
        (Tensor str): The true label of the image.
        
    Exception:
        If eager execution is disabled.
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
        ".png|.jpg|.jpeg|.bmp", 
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
    
    # Crop the image TODO add scaling/resize/reshape?
    if do_cropping:
        try:
            image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
        except:
            raise ValueError(colored('Cropping bounds are invalid. Please review target size and cropping position values.', 'red'))
    
    # Find the label if needed
    if use_labels:
        if not label_position or label_position < 0:
            raise ValueError(colored("Error: A label position is needed to parse the image class.", 'red'))
        
        # Check for label matches
        path_label = tf.strings.split(path_substring, "_")[label_position]
        class_matches = np.where(path_label == class_names)[0]
        if len(class_matches) > 1:
            raise ValueError(colored(f'Error: more than one class name found in the image: "{image_path}"', 'red'))
        elif len(class_matches) < 1:
            raise ValueError(colored(f'Error: no class name found was in the image: "{image_path}"\n\tIs the case correct?', 'red'))
        
        # Return the tensor
        return image, class_matches[0]
    else:
        return image
    



""" --- WIP --------------------------------------------------------------------- """


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

    def parse_image(self, filename,class_names, channels, do_cropping, offset_height, offset_width, target_height, target_width, label_position=None, use_labels=True): 
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

    def parse_image(self, filename, class_names, channels, do_cropping, offset_height, offset_width, target_height, target_width, label_position=None, use_labels=True): 
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
        
        return image, tf.argmax(label_bool)
    