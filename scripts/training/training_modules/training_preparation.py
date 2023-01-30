from training.training_modules.fold_generator import generate_folds
from training.training_modules.index_getter import get_indexes
from training.training_modules.image_getter import get_files
from random import seed


class TrainingVars:
    def __init__(self, config, test_subject):
        """ Gets some parameters needing for training
            
        Args:
            config (dict): List of input image paths.
            test_subject (str): List of class names.
        """
        # Set the seed
        seed(config['seed'])
        
        # The files to train with and info about its contents
        self.files = get_files(config['data_input_directory'])
        self.indexes, self.label_position = get_indexes(self.files, config['class_names'], config['validation_subjects'])
        
        # Generate training folds
        self.folds, self.n_folds = generate_folds(config['test_subjects'], config['validation_subjects'], test_subject, config['shuffle_the_folds'])
        
