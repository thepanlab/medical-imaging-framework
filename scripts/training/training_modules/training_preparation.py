from training.training_modules.fold_generator import generate_folds
from training.training_modules.index_getter import get_indexes
from training.training_modules.image_getter import get_files
from random import seed


class TrainingVars:
    def __init__(self, config, test_subject):
        """ Gets some parameters needing for training
            
            -- Input Parameters ------------------------
            config (dict): List of input image paths.
            test_subject (str): List of class names.
            --------------------------------------------
            
            -- Variables -------------------------------
            files (list of str): A list of file paths.
            folds (list of dict): A list of fold partitions.
            
            n_folds (int): The number of folds to train with.
            indexes (dict of lists): A dictionary of lists containing various indexes.
            label_position (int): Position in the file name where the label exists.
            --------------------------------------------
        """
        # Set the seed
        seed(config['seed'])
        
        # Let the subjects be of the same case
        config['subject_list'] = [s.lower() for s in config['subject_list']]
        
        # The files to train with and info about its contents
        self.files = get_files(config['data_input_directory'])
        self.indexes, self.label_position = get_indexes(self.files, config['class_names'], config['subject_list'])
        
        # Generate training folds
        self.folds, self.n_folds = generate_folds(config['subject_list'], test_subject, config['n_folds'], config['shuffle_the_folds'])
        