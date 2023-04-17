from training.training_modules.data_processing.fold_generator import generate_folds
from training.training_modules.data_processing.index_getter import get_indexes
from training.training_modules.image_processing.image_getter import get_files
from random import seed


class TrainingVars:
    def __init__(self, config, test_subject, is_outer, training_subject=None):
        """ Gets some parameters needing for training
            
        Args:
            config (dict): List of input image paths.
            test_subject (str): The test subject to train.
            is_outer (bool): A flag telling if running the outer loop.
        """
        # Set the seed
        seed(config['seed'])
        
        # The files to train with and info about its contents
        self.files = get_files(config['data_input_directory'], config['shuffle_the_images'], config['seed'])
        self.indexes, self.label_position = get_indexes(self.files, config['class_names'], config['subject_list'])
        
        # Make sure test subjects and validation subjects are unique
        test_subjects = list(set(config['test_subjects']))
        if is_outer:
            validation_subjects = None
        else:
            validation_subjects = list(set(config['validation_subjects']))
    
        # Generate training folds
        self.folds, self.n_folds = generate_folds(test_subjects, validation_subjects, config['subject_list'], test_subject, config['shuffle_the_folds'], training_subject=training_subject)
            