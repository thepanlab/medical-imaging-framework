from training.training_modules.result_outputter import output_results
from training.training_modules.model_creator import TrainingModel
from training.training_modules.image_parser import parse_image
from training.training_modules.training_fold import Fold
from training.checkpoint_modules.logger import *
from termcolor import colored
from time import perf_counter
from tensorflow import keras
import tensorflow as tf


def training_loop(config, testing_subject, files, folds, rotations, indexes, label_position):
    """ Creates a model, trains it, and writes its outputs.
        
        -- Input Parameters ------------------------
        config (dict): The input configuration.
        
        testing_subject (str): The name of the testing subject.
        files (list of str): A list of filepaths to images.
        folds (list of dict): A list of fold partitions.
        
        rotations (int): the number of rotations to perform.
        indexes (dict of lists): A list of true indexes.
        label_position (int): Location in the filename of the label.
        --------------------------------------------
    """
    print(colored(f'Beginning the training loop for {testing_subject}.', 'green'))
       
    # Get the current rotation progress
    rotation = 0
    log_rotations = read_log_items(
        config['output_path'], 
        config['job_name'], 
        ['current_rotation']
    )
    if log_rotations and 'current_rotation' in log_rotations:
        rotation = log_rotations['current_rotation']
    
    # Train for every rotation specified
    for rot in range(rotation, rotations):
        print(colored(f'-- Rotation {rot+1}/{rotations} for {testing_subject} ---------------------------------------', 'magenta'))
        training_fold = Fold(rot, config, testing_subject, files, folds, indexes, label_position)
        training_fold.run_all_steps()
        
        # Write the index to log
        write_log(
            config['output_path'], 
            config['job_name'], 
            {'current_rotation': rot + 1}
        )
        
    # Reset the rotations
    write_log(
        config['output_path'], 
        config['job_name'], 
        {'current_rotation': 0}
    )       

    