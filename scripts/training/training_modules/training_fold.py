from training.training_modules.result_outputter import output_results
from training.training_modules.model_creator import TrainingModel
from training.checkpoint_modules.checkpointer import *
from training.training_modules.image_parser import *
from training.checkpoint_modules.logger import *
from termcolor import colored
from time import perf_counter
from tensorflow import keras
import tensorflow as tf


class _FoldTrainingInfo():
    def __init__(self, fold_index, config, testing_subject, files, folds, indexes, label_position):
        """ Initializes a training fold info object.

        Args:
            fold_index (int): The fold index within the loop.
            config (dict): The training configuration.
            test_subject (str): The test subject name.
            files (list of str): A list of filepaths to images.
            folds (list of dict): A list of fold partitions
            indexes (dict of lists): A list of true indexes.
            label_position (int): Location in the filename of the label.
        """ 
        self.datasets = {
            'testing':    {'files': [], 'indexes': [], 'labels': [], 'ds': None},
            'training':   {'files': [], 'indexes': [], 'labels': [], 'ds': None},
            'validation': {'files': [], 'indexes': [], 'labels': [], 'ds': None}
        }
        self.testing_subject = testing_subject
        self.validation_subject = folds[fold_index]['validation'][0]
        
        self.files = files
        self.folds = folds
        self.config = config
        self.indexes = indexes
        self.fold_index = fold_index
        self.label_position = label_position
        
        self.model = None
        self.callbacks = None
        self.checkpoint_name = None
        
        
    def run_all_steps(self):
        """ This will run all of the steps in order to create the basic training information. """
        self.get_dataset_info()
        self.create_model()
        self.create_callbacks()
        
        
    def get_dataset_info(self):
        """ Get the basic data to create the datasets from.
            This includes the testing and validation subjects, file paths,
            label indexes, and labels.
        """
        for index, file in enumerate(self.files):
            dataset = ''
            if self.indexes['subjects'][index] == self.validation_subject:
                dataset = 'validation'
            elif self.indexes['subjects'][index] == self.testing_subject:
                dataset = 'testing'
            else:
                dataset = 'training'
            self.datasets[dataset]['files'].append(file)
            self.datasets[dataset]['indexes'].append(index)
            self.datasets[dataset]['labels'].append(self.indexes['labels'][index])
            
            
    def create_model(self):  
        """ Create the initial model for training. """
        self.model = TrainingModel(
            self.config['hyperparameters'],
            self.config['selected_model_name'], 
            self.config['target_height'], 
            self.config['target_width'], 
            self.config['class_names']
        )
        
    
    def create_callbacks(self):
        """ Create the training callbacks. 
            This includes early stopping and checkpoints.
        """
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=self.config['hyperparameters']['patience'], 
            restore_best_weights=True
        )
        
        self.checkpoint_prefix = f"{self.config['job_name']}_config_{self.config['selected_model_name']}_test_{self.testing_subject}_val_{self.validation_subject}"     
        
        checkpoints = Checkpointer(
            self.config['hyperparameters']['epochs'],
            self.config['k_epoch_checkpoint_frequency'], 
            self.checkpoint_prefix, 
            self.config['output_path']
        )
        self.callbacks = (early_stopping, checkpoints)
        


class Fold():
    def __init__(self, fold_index, config, testing_subject, files, folds, indexes, label_position):
        """ Initializes a training fold object.

        Args:
            fold_index (int): The fold index within the loop.
            config (dict): The training configuration.
            test_subject (str): The test subject name.
            files (list of str): A list of filepaths to images.
            folds (list of dict): A list of fold partitions
            indexes (dict of lists): A list of true indexes.
            label_position (int): Location in the filename of the label.
        """ 
        self.fold_info = _FoldTrainingInfo(
            fold_index, 
            config, 
            testing_subject, 
            files, 
            folds, 
            indexes, 
            label_position
        )
        self.history = None
        self.time_elapsed = None
        self.checkpoint_epoch = 0
        
        
    def run_all_steps(self):
        """ This will run all of the steps for the training process 
            Training itself depends on the state of the training fold.
            It checks for insufficient dataset.
        """
        # Load in the previously saved fold info. Check if valid. If so, use it.
        prev_info = self.load_state()
        if prev_info is not None and \
        self.fold_info.testing_subject == prev_info.testing_subject and \
        self.fold_info.validation_subject == prev_info.validation_subject:
            self.fold_info = prev_info
            self.load_checkpoint()
            print(colored("Loaded previous existing state for test. " + 
                          f"{prev_info.testing_subject} and val. {prev_info.validation_subject}.", 'cyan'))

        else:
            self.fold_info.run_all_steps()
            self.save_state()
            
        # Create the datasets and train. (Datasets cannot be logged.)
        if self.create_dataset():
            self.train_model()
            self.output_results()
    
    
    def load_state(self):
        """ Loads the latest training state. """
        log = read_log_items(
            self.fold_info.config['output_path'], 
            self.fold_info.config['job_name'], 
            ['fold_info']
        )
        if log is None or 'fold_info' not in log:
            return None
        return log['fold_info']
            
    
    def save_state(self):
        """ Saves the state of the fold to a log. """
        write_log(
            self.fold_info.config['output_path'], 
            self.fold_info.config['job_name'], 
            {'fold_info': self.fold_info}
        )
    
    
    def load_checkpoint(self):
        """ Loads the latest checkpoint to start from. """
        results = get_most_recent_checkpoint(
            self.fold_info.config['output_path'], 
            self.fold_info.checkpoint_prefix
        )
        if results is not None:
            print(colored(f"Loaded most recent checkpoint of epoch: {results[1]}.", 'cyan'))
            self.fold_info.model.model = results[0]
            self.checkpoint_epoch = results[1]
        
    def create_dataset(self):
        """ Create the dataset needed for training the model.
            It will map the image paths into their respective image and label pairs.
            TODO Complete any image-reading changes here for different file types.
        """
        # Get the datasets for each phase
        for dataset in self.fold_info.datasets:
            if not self.fold_info.datasets[dataset]['files']:
                continue
            
            # Create the special image type readers
            csvreader = ImageReaderCSV(configs=self.fold_info.config)
            imreader = ImageReaderGlobal()
            
            # Parse images here
            ds = tf.data.Dataset.from_tensor_slices(self.fold_info.datasets[dataset]['files'])
            ds_map = ds.map(lambda x: tf.py_function(
                func=parse_image,
                inp=[
                    x,                                                                 # Filename
                    self.fold_info.config['class_names'],                              # Class Names
                    self.fold_info.config['hyperparameters']['channels'],              # Channels
                    self.fold_info.config['hyperparameters']['do_cropping'],           # Do Cropping
                    self.fold_info.config['hyperparameters']['cropping_position'][0],  # Offset Height
                    self.fold_info.config['hyperparameters']['cropping_position'][1],  # Offset Width
                    self.fold_info.config['target_height'],                            # Target Height
                    self.fold_info.config['target_width'],                             # Target Width
                    self.fold_info.label_position,                                     # Label Position
                ],
                Tout=[tf.float32, tf.int64]
            ))
            self.fold_info.datasets[dataset]['ds'] = ds_map.batch(self.fold_info.config['hyperparameters']['batch_size'], drop_remainder=False)
            
        # If the datasets are empty, cannot train
        if self.fold_info.datasets['validation']['ds'] is None or self.fold_info.datasets['training']['ds'] is None:
            print(colored(
                f"Non-fatal Error: training was skipped for the Test Subject {self.fold_info.testing_subject} and Validation Subject {self.fold_info.validation_subject}. " + 
                f"There were no files in the training and/or validation datasets.\n",
                'yellow'
            ))
            return False
        return True 
        
        
    def train_model(self):
        """ Train the model, assuming the given dataset is valid. """  
        if self.checkpoint_epoch+1 == self.fold_info.config['hyperparameters']['epochs']:
            print(colored("Maximum number of epochs reached from checkpoint.", 'yellow'))
            return
        time_start = perf_counter()
        self.history = self.fold_info.model.model.fit(
            self.fold_info.datasets['training']['ds'],
            validation_data=self.fold_info.datasets['validation']['ds'],
            epochs=self.fold_info.config['hyperparameters']['epochs'],
            initial_epoch=self.checkpoint_epoch,
            callbacks=[self.fold_info.callbacks]
        )
        self.time_elapsed = perf_counter() - time_start
        
        
        
    def output_results(self):
        """ Output the training results to file. """
        print(colored(f"Finished training for testing subject {self.fold_info.testing_subject} and validation subject {self.fold_info.validation_subject}.", 'green'))
        output_results(
            self.fold_info.config['output_path'], 
            self.fold_info.testing_subject, 
            self.fold_info.validation_subject, 
            self.fold_info.fold_index, 
            self.fold_info.model, 
            self.history, 
            self.time_elapsed, 
            self.fold_info.datasets, 
            self.fold_info.config['class_names']
        )
        