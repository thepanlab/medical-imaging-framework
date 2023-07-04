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
    
    # Get the needed data for training
    training_vars = training_preparation.TrainingVars(config, is_outer, test_subject)
    
    # Start training loop
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
      
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # https://stackoverflow.com/questions/66768279/running-tf-config-run-functions-eagerlytrue-in-python-3
    # Disables eager execution.
    # tf.compat.v1.disable_eager_execution()

    # Enables eager execution of tf.functions.
    # tf.config.run_functions_eagerly(True)
    
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
        if len(set(config["validation_subjects"])) != len(config["validation_subjects"]):
            raise ValueError("You have repeated validation_subjects!. Please verify your list of validation subjects.")
        
        test_subjects = list(test_subjects)
        
        if is_outer == False:
            if len(set(test_subjects)) != len(test_subjects):
                raise ValueError("You have repeated test_subjects!. Please verify your list of test subjects.")
            
            
        # Train for each test subject
        for test_subject in test_subjects:
            subject_loop(config, test_subject, is_outer)
            write_log(
                config['output_path'], 
                config['job_name'], 
                {'test_subjects': [t for t in test_subjects if t != test_subject]},
                use_lock=False
            )