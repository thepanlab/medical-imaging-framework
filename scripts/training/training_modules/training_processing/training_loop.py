from training.training_modules.training_processing.training_fold import Fold
from training.training_checkpointing_logging.logger import *
from termcolor import colored


def training_loop(config, testing_subject, files, folds, rotations, indexes, label_position, is_outer, rank=None):
    """ Creates a model, trains it, and writes its outputs.
        
    Args:
        config (dict): The input configuration.
        
        testing_subject (str): The name of the testing subject.
        files (list of str): A list of filepaths to images.
        folds (list of dict): A list of fold partitions.
        
        rotations (int): the number of rotations to perform.
        indexes (dict of lists): A list of true indexes.
        label_position (int): Location in the filename of the label.
        
        is_outer (bool): If this is of the outer loop. Default is false. (Optional)
        rank (int): The process rank. Default is none. (Optional)
    """
    print(colored(f'Beginning the training loop for {testing_subject}.', 'green'))
       
    # Get the current rotation progress
    rotation = 0
    log_rotations = read_log_items(
        config['output_path'], 
        config['job_name'], 
        ['current_rotation'],
        rank
    )
    
    if log_rotations and 'current_rotation' in log_rotations and testing_subject in log_rotations['current_rotation']:
        rotation = log_rotations['current_rotation'][testing_subject]
        print(colored(f'Starting off from rotation {rotation+1} for testing subject {testing_subject}.', 'cyan'))
    
    # Train for every rotation specified
    for rot in range(rotation, rotations):
        if is_outer:
            rot_subject = folds[rot]['training'][0]
            print(colored(f'--- Rotation {rot+1}/{rotations} for test subject {testing_subject} ---', 'magenta'))
        
        # Inner loop will use the validation subjects
        else:
            rot_subject = folds[rot]['validation'][0]
            print(colored(f'--- Rotation {rot+1}/{rotations} for test subject {testing_subject} and val subject {rot_subject} ---', 'magenta'))
        
        # Create and run the training fold for this subject pair
        training_fold = Fold(rot, config, testing_subject, rot_subject, files, folds, indexes, label_position, rank, is_outer)
        training_fold.run_all_steps()
        
        # Write the index to log
        if not log_rotations:
            rotation_dict = {testing_subject: rot + 1}
        else:
            rotation_dict = log_rotations['current_rotation']
            rotation_dict[testing_subject] = rot + 1
        write_log(
            config['output_path'], 
            config['job_name'], 
            {'current_rotation': rotation_dict},
            rank
        )
             

    