from abc import ABC, abstractmethod
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
        
        label_position (int): Optional. The position of the label in the image name. Default is None.
        use_labels (bool): Optional. Whether to consider/output the true image label. Default is True.
        
    Returns:
        (Tensor image): An image.
        (Tensor str): The true label of the image.
        
    Exception:
        If eager execution is disabled.
    """
    
    # Split to get only the image name
    image_path = tf.strings.split(filename, "/")[-1]
    
    # Remove the file extention
    path_substring = tf.strings.regex_replace(
        image_path,
        ".png|.jpg|.jpeg|.tiff|.csv", 
        ""
    )

    def io_read(file):
        res = io.imread(file.numpy().decode())
        res_to_return=res.reshape(185, 210, 185)
        return res_to_return

    #print("shape print : ", image.shape)
    image = np.array(tf.py_function(io_read, [filename], [tf.float32]))[0]
    #image = tf.io.decode_image(image, channels=channels, dtype=tf.float32, name=None, expand_animations=False)
    
    # TODO: Perhaps make image parser one function, so it can be changed in one area?
    # image_type = tf.strings.split(filename, ".")[-1].numpy().decode()
    # if image_type == 'csv':
        # TODO: do this
    # else:
        # TODO: do that
    
    # Crop the image
    if do_cropping:
        image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
    
    # Find the label if needed
    if use_labels:
        path_label = tf.strings.split(path_substring, "_")[label_position]
        class_name = tf.argmax(path_label == class_names)
        return image, class_name
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
    