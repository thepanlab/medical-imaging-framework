from training.training_modules import training_preparation
from training.training_modules import console_printing
from training.training_modules import training_loop
from util.get_config import parse_training_configs
from training.checkpoint_modules.logger import *
from termcolor import colored
import tensorflow as tf

# Location of the configurations
CONFIG_LOC = './training/training_config_files'
            
            
def subject_loop(config, test_subject):
    """ Executes the training loop for the given test subject.

    Args:
        config (dict): The training configuration.
        test_subject (str): The test subject name.
    """
    print(colored(
        f"\n\n===========================================================\n" + 
        f"Starting training for {test_subject} in {config['selected_model_name']}\n"
        , 'magenta'
    ))
    training_vars = training_preparation.TrainingVars(config, test_subject)
    training_loop.training_loop(
        config, 
        test_subject, 
        training_vars.files, 
        training_vars.folds, 
        training_vars.n_folds, 
        training_vars.indexes, 
        training_vars.label_position
    )


def main():
    """ Runs the training process for each configuration and test subject. """
    # Print a list of the available GPUs
    console_printing.show_gpu_list()
    tf.config.run_functions_eagerly(True)
    
    # Parse the command line arguments
    configs = parse_training_configs(CONFIG_LOC)
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
            
        # Train for each test subject
        for test_subject in test_subjects:
            subject_loop(config, test_subject)
            test_subjects.remove(test_subject)
            write_log(
                config['output_path'], 
                config['job_name'], 
                {'test_subjects': test_subjects}
            )

if __name__ == "__main__":
    """ Called when this file is run. """
    main()
