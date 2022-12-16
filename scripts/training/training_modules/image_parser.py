from tensorflow import keras as K
import tensorflow as tf

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
        use_labels(bool): Optional. Whether to consider/output the true image label. Default is True.
        --------------------------------------------
        
        -- Return ----------------------------------
        (Tensor image): An image.
        (Tensor str): The true label of the image.
        --------------------------------------------
    """
    # Split to get only the image name
    image_path = tf.strings.split(filename, "/")[-1]
    
    # Remove the file extention
    path_substring = tf.strings.regex_replace(
        image_path,
        ".png|.jpg|.jpeg", 
        ""
    )
    
    # Read the image --> TODO modify this for 3D images
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels=channels, dtype=tf.float32, name=None, expand_animations=False)
    
    # Crop the image
    if do_cropping == 'true':
       image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
       
    # Normalize the image
    if use_mean == 'true':
        image = image - mean / 255
    
    # Find the label if needed
    if use_labels:
        label = tf.strings.split(path_substring, "_")[label_position]
        label_bool = (label == class_names)
        return image, tf.argmax(label_bool)
    else:
        return image
