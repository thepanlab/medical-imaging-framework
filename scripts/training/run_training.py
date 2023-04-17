from training.training_modules.data_processing import training_preparation
from training.training_modules.output_processing import console_printing
from training.training_modules.training_processing import training_loop
from training.training_checkpointing_logging.logger import *
from util.get_config import parse_training_configs
from termcolor import colored
import tensorflow as tf
            
            
def subject_loop(config, test_subject, is_outer):
    """ Executes the training loop for the given test subject.

    Args:
        config (dict): The training configuration.
        test_subject (str): The test subject name.
        is_outer (bool): If this is of the outer loop.
    """
    print(colored(
        f"\n\n===========================================================\n" + 
        f"Starting training for {test_subject} in {config['selected_model_name']}\n"
        , 'magenta'
    ))
    training_vars = training_preparation.TrainingVars(config, test_subject, is_outer)
    training_loop.training_loop(
        config, 
        test_subject, 
        training_vars.files, 
        training_vars.folds, 
        training_vars.n_folds, 
        training_vars.indexes, 
        training_vars.label_position,
        is_outer
    )


def main(config_loc, is_outer):
    """ Runs the training process for each configuration and test subject.

    Args:
        is_outer (bool): If this is of the outer loop.
    """
    # Print a list of the available GPUs
    gpus = console_printing.show_gpu_list()
    
    # Set eager execution and memory limits
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=tf_config)
    tf.config.run_functions_eagerly(True)
    
    # Parse the command line arguments
    configs = parse_training_configs(config_loc)
    for config in configs:
        
        # Read in the log's subject list, if it exists
        log_list = read_log_items(
            config['output_path'], 
            config['job_name'], 
            ['test_subjects']
        )
        if log_list and 'subject_list' in log_list:
            test_subjects = log_list['test_subjects']
        else:
            test_subjects = config['test_subjects']
            
        # Double-check that the test subjects are unique
        test_subjects = list(set(test_subjects))
            
        # Train for each test subject
        for test_subject in test_subjects:
            subject_loop(config, test_subject, is_outer)
            write_log(
                config['output_path'], 
                config['job_name'], 
                {'test_subjects': [t for t in test_subjects if t != test_subject]}
            )