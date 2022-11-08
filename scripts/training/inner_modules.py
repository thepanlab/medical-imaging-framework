import pickle
import os

def get_variables(data):
    """ Returns needed variables for training. Kept separate to save space.

    Args:
        data (dict): The json configuration.
    """
    batch_size = int(data['batch_size'])                        # (int)
    n_epochs = int(data['epochs'])                              # (int)
    subject_list = data['subject_list'].split(',')              # (string list)
    subject_list = [x.lower() for x in subject_list]            # (string list lower case)
    file_path = data['files_directory']                         # main directory path (string)
    results_path = data['results_path']                         # (string)
    checkpoints_path = data['checkpoints_path']                 # (string)
    checkpoint_title = data['checkpoint_title']                 # (string)
    
    the_seed = int(data['seed'])                                # (int)
    class_names = data['classes_names'].split(',')              # (string list)
    learning_rate = float(data['learning_rate'])                # (float)
    the_momentum = float(data['momentum'])                      # (float)
    the_decay = float(data['decay'])                            # (float)
    
    the_patience = int(data['patience'])                        # (int)
    channels = int(data['channels'])                            # (int)
    shuffle_the_folds = data['shuffle_the_folds']               # (bool)
    mean = float(data['mean'])                                  # (int)
    use_mean = data['use_mean']                                 # (string)
    
    cropping_position = data['cropping_position'].split(',')    # (string list)
    image_size = data['image_size'].split(",")                  # (string list)
    offset_height = int(cropping_position[0])                   # (int)
    offset_width = int(cropping_position[1])                    # (int)
    target_height = int(data['target_height'])                  # (int)
    target_width = int(data['target_width'])                    # (int)
    do_cropping = data['do_cropping']                           # (string)
    
    selected_model = data['selected_model']                     # (string)
    rotations_config = data['rotations']                        # (string)
    rotations = 0                                               # (int)
    label_position = -1                                         # (int)
    
    return batch_size, n_epochs, subject_list, file_path, results_path, checkpoints_path, checkpoint_title, \
        the_seed, class_names, learning_rate, the_momentum, the_decay, \
        the_patience, channels, shuffle_the_folds, mean, use_mean, \
        cropping_position, image_size, offset_height, offset_width, \
        target_height, target_width, do_cropping, \
        selected_model, rotations_config, rotations, label_position
        

def create_folders(path, names=None):
    """ Creates folder(s) if they do not exist.

    Args:
        path (str): Name of the path to the folder.
        names (list): The folder name(s). (Optional.)
    """
    if names is not None:
        for name in names:
            folder_path = os.path.join(path, name)
            if not os.path.exists(folder_path): os.makedirs(folder_path)
    else:
        if not os.path.exists(path): os.makedirs(path)
            

def check_checkpoints(checkpoints_path, checkpoint_title):
    """ This will check if checkpointing exists for this title.

    Args:
        checkpoints_path (str): Path to where all checkpoints are written to.
        checkpoint_title (str): The title of the checkpointing session.

    Returns:
        bool: True if checkpoints exists, else false.
    """
    # If this checkpoint-title doesn't exist, create new checkpoints
    target_folder = os.path.join(checkpoints_path, checkpoint_title)
    if not os.path.exists(target_folder): 
        os.makedirs(target_folder)
        return False
    
    # If there is no log file, create new checkpoints
    target_file = os.path.join(target_folder, f'{checkpoint_title}_state.log')
    if not os.path.exists(target_file): 
        return False
    
    # Else, we need to build off of pre-existing checkpoints
    return True


def save_state(checkpoints_path, checkpoint_title, data):
    """ Saves a tuple of variables to a file.

    Args:
        checkpoints_path (str): Path to where all checkpoints are written to.
        checkpoint_title (str): The title of the checkpointing session.
        data (tuple): A tuple of various items.
    """
    target_file = os.path.join(checkpoints_path, checkpoint_title, f'{checkpoint_title}_state.log')
    with open(target_file, 'wb') as fp:
        pickle.dump(data, fp)
    
    
def load_previous_state(checkpoints_path, checkpoint_title):
    """ Loads the most recent state of some checkpointing process.

    Args:
        checkpoints_path (str): Path to where all checkpoints are written to.
        checkpoint_title (str): The title of the checkpointing session.

    Returns:
        tuple: A tuple of various variables.
    """
    target_file = os.path.join(checkpoints_path, checkpoint_title, f'{checkpoint_title}_state.log')
    with open(target_file, 'rb') as fp:
        return pickle.load(fp)

