import tensorflow as tf

from tensorflow import keras as K

def parse_image(filename, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width): 
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
    # Split to get only the image name
    image_path = tf.strings.split(filename, "/")[-1]
    
    # Remove the file extention
    path_substring = tf.strings.regex_replace(
        image_path,
        ".png|.jpg|.jpeg", 
        ""
    )
    
    # Find the label
    label = tf.strings.split(path_substring, "_")[label_position]
    label_bool = (label == class_names)
    
    # Read the image --> TODO modify this for 3D images
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels=channels, dtype=tf.float32, name=None, expand_animations=False)
    
    # Crop the image
    if do_cropping == 'true':
       image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
       
    # Normalize the image
    if use_mean == 'true':
        image = image - mean / 255
    return image, tf.argmax(label_bool)
