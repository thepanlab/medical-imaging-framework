from training.training_modules import training_preparation
from training.training_modules import console_printing
from training.training_modules import training_loop
from util.get_config import parse_training_configs
from training.checkpoint_modules.logger import *
from termcolor import colored
import tensorflow as tf
import shutil


# When debugging the logs/ckpts, enable to delete old logs/ckpts
DEBUG_LOGS = True

# Location of the configurations
CONFIG_LOC = './training/training_config_files'
            
            
def subject_loop(config, test_subject):
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
    configs = parse_training_configs('./training/training_config_files')
    for config in configs:
    
        # For debugging logging
        if DEBUG_LOGS:
            log_path = os.path.join(config['output_path'], 'logging')
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
        
        # Make sure the subject list is of the same case
        subject_list = [s.lower() for s in config['subject_list']]
        
        # Read in the log's subject list, if it exists
        log_list = read_log_items(
            config['output_path'], 
            config['job_name'], 
            ['subject_list']
        )
        if log_list and 'subject_list' in log_list:
            subject_list = log_list['subject_list']
            
        # Train for each test subject
        for test_subject in subject_list:
            subject_loop(config, test_subject)
            subject_list.remove(test_subject)
            write_log(
                config['output_path'], 
                config['job_name'], 
                {'subject_list': subject_list}
            )

if __name__ == "__main__":
    """ Called when this file is run. """
    main()
