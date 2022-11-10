from termcolor import colored
from random import shuffle
import os


def get_files(input_path):
    """ Gets all of the input paths of the image data.
        
        -- Input Parameters ------------------------
        input_path (str): A path to some directory.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of str): A list of image paths.
        --------------------------------------------
    """
    # See if it exists
    if not os.path.isdir(input_path):
        raise Exception(colored(f"Error: '{input_path}' is not a valid input path.", 'red'))
    
    # Search through the first level, of subdirectories (1)
    files = []
    _flatten_dir(input_path, files)
                    
    # Shuffle the list
    files.sort()
    shuffle(files)
    print(colored('Finished getting the image paths.', 'green'))
    return files

    
def _flatten_dir(path, files):
    """ Recursively gets the paths of ALL images within a directory and its subdirectories.

        -- Input Parameters ------------------------
        path (str): A path to some directory.
        files (list of str): A list of paths to images.
        --------------------------------------------
    """
    for item in os.listdir(os.path.abspath(path)):
        full_path = os.path.join(path, item)
        if os.path.isfile(full_path):
            if full_path.endswith((".png", ".jpg", ".jpeg")):
                files.append(full_path)
            else:
                print(colored(f"Warning: Non-image file detected: '{full_path}'"))
        else:
            _flatten_dir(full_path, files)
 