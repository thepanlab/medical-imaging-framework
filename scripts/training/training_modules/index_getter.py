from termcolor import colored


def get_indexes(files, class_names, subject_list):
    """ Gets the indexes of classes, and the subjects within the image file names.
        
    Args:
        files (list of str): List of input image paths.
        class_names (list of str): List of class names.
        subject_list (list of str): List of subject names
        
    Returns:
        (dict of lists): A dictionary containing all labels, indexes, and subjects.
        (int): The label position.
        
    Exception:
        When more than one label or subject is given.
    """
    indexes = {'labels': [], 'idx': [], 'subjects': []}
    label_position = None
    
    for file in files:
        
        # Get the proper filename to parse
        formatted_name = file.replace("%20", " ").split('/')[-1].split('.')[:-1]
        formatted_name = ''.join(formatted_name)
        formatted_name = formatted_name.split('_')
        
        # Get the image label
        labels = [c for c in class_names if c in formatted_name]
        if len(labels) != 1:
            raise ValueError(colored(f"Error: {len(labels)} labels found for '{file}'. \nThere should be exactly one. Is the case correct?", 'red'))
        
        # Get index of the label in the class name list
        idx = class_names.index(labels[0])
            
        # Get the image subject
        subjects = [s for s in subject_list if s in formatted_name]
        if len(subjects) != 1:
            raise Exception(colored(f"Error: {len(subjects)} subjects found for '{file}'. \nThere should be exactly one. Is the case correct?", 'red'))
        
        # Get the position of the label in the string
        if not label_position:
            label_position = formatted_name.index(labels[0])
            
        # Add the values
        indexes['labels'].append(labels[0])
        indexes['idx'].append(idx)
        indexes['subjects'].append(subjects[0])
        
    print(colored('Finished finding the label indexes.', 'green'))
    return indexes, label_position
 