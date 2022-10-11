from termcolor import colored
import tensorflow as tf
import os

def parse_image(filename, mean, use_mean, class_names, label_position, channels, 
                do_cropping, offset_height, offset_width, target_height, target_width):
    parts_0 = tf.strings.split(filename, ".")
    complete = parts_0[0]
    if len(parts_0) > 2:
        for part in parts_0[1:-1]:
            complete = tf.add(complete,part)
    parts = tf.strings.split(complete, "_")
    label = parts[label_position]
    label_bool = (label == class_names)
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels=channels, dtype=tf.float32,
                               name=None, expand_animations=False)
    if do_cropping == 'true':
       image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
    if use_mean == 'true':
        image = image - mean / 255
    return image, tf.argmax(label_bool)


def get_label_subject(path,label_position ,class_names, subject_list):
    formatted_path=path.lower().replace("%20", " ")
    
    # Get all match the labels
    labels = [class_name for class_name in class_names if class_name in formatted_path]
    
    # Assuming only 1 label will be obtained, otherwise throw exception
    if len(labels) > 1:
        raise Exception(colored(f"Error: Duplicate labels extracted from: {path}", 'red'))
    elif len(labels) < 1:
        raise Exception(colored(f"Error: Could not get label from: {path}", 'red'))

    # Update label index within the filename, this line only perform once
    if label_position==-1:
        temp = os.path.abspath(formatted_path).split('.')
        label_position=temp[0].split('_').index(labels[0])
    idx = class_names.index(labels[0])
    
    # Get all match the subjects
    subjects = [subject for subject in subject_list if subject in formatted_path]
    
    # Assuming only 1 subject will be obtained, otherwise throw exception
    if len(subjects) > 1:
        raise Exception(colored(f"Duplicate subjects extracted from: {path}"))
    elif len(subjects) < 1:
        raise Exception(colored(f"Error when getting subject from: {path}"))
    return labels[0], idx, subjects[0], label_position
