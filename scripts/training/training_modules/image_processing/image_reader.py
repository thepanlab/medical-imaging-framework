from abc import ABCMeta, abstractmethod
from tensorflow import keras as K
import tensorflow as tf
from skimage import io
import numpy as np


class ImageReader(metaclass=ABCMeta):
    def __init__(self):
        return
    
    @abstractmethod
    def io_read(self, filename):
        #return io.imread(filename.numpy().decode())
        pass

    @abstractmethod
    def parse_image(self, filename, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width):
        pass



class ImageReaderGlobal(ImageReader):
    def __init__(self):
        ImageReader.__init__(self)
        return

    def io_read(self, filename):
        return

    def parse_image(self, filename, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width):
        return


class ImageReaderCSV(ImageReader):
    def __init__(self):
        ImageReader.__init__(self)
        return

    def io_read(self, filename):
        image = np.genfromtxt(filename)
        image.reshape()
        return

    def parse_image(self, filename, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width):
        return