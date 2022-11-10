from training.training_modules import training_preparation
from training.training_modules import console_printing
from training.training_modules import training_loop
from util.get_config import parse_training_configs
from termcolor import colored


def main():
    """ Runs the training process for each configuration and test subject. """
    # Print a list of the available GPUs
    console_printing.show_gpu_list()
    
    # Parse the command line arguments
    configs = parse_training_configs('./training/training_config_files')
    for config in configs:
        
        # Make sure the subject list is of the same case
        config['subject_list'] = [s.lower() for s in config['subject_list']]
        
        # Train for each test subject
        for test_subject in config['subject_list']:
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
                training_vars.rotations, 
                training_vars.indexes, 
                training_vars.label_position
            )
    

if __name__ == "__main__":
    """ Called when this file is run. """
    main()
