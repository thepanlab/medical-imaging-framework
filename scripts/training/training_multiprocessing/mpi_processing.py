from training.training_modules.data_processing import training_preparation, fold_generator
from training.training_modules.output_processing import console_printing
from training.training_modules.training_processing import training_loop
from training.training_checkpointing_logging.logger import *
from util.get_config import parse_training_configs
from termcolor import colored
from datetime import date
import tensorflow as tf
from mpi4py import MPI
import datetime
import time
import math
import os

# Location of the configurations
CONFIG_LOC = './training/training_config_files'

def split_tasks(configs, n_proc, is_outer):
    """ Generates config-fold tuples for training.

        -- Input Parameters ------------------------
        configs (list of dict): List of configurations.
        n_proc (int): Number of training processes.
        is_outer (bool): If this is of the outer loop or not.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list tuples): A list of config-fold tuples.
        --------------------------------------------
    """
    # Create a list of (config, test subject, validation subject) tuples
    tasks = []
    for config in configs:
        
        # Generate all fold-pairs
        test_subjects = config['test_subjects']
        validation_subjects = None if is_outer else config['validation_subjects']
        folds = fold_generator.generate_pairs(test_subjects, validation_subjects, config['subject_list'], config['shuffle_the_folds'])
        
        # Add folds to task list
        tasks.extend([(config, test_subject, training_subject) for test_subject, training_subject in folds])
        
    return tasks
        
            
def run_training(rank, config, test_subject, training_subject, is_outer):
    
    # Read in the log, if it exists
    job_name = f"{config['job_name']}_test_{test_subject}" if is_outer else f"{config['job_name']}_test_{test_subject}_sub_{training_subject}"
    log = read_log_items(
        config['output_path'], 
        job_name, 
        ['is_finished']
    )
    
    # If this fold has finished training, return
    if log and log['is_finished']:
        return
    
    # Run the subject pair
    if is_outer:
        print(colored(f"Rank {rank} is starting training for {test_subject}.", 'green'))
    else:
        print(colored(f"Rank {rank} is starting training for {test_subject} and validation subject {training_subject}.", 'green'))
    subject_loop(rank, config, test_subject, is_outer, training_subject=training_subject)
    write_log(
        config['output_path'], 
        job_name, 
        {'is_finished': True}
    )
        
        
def subject_loop(rank, config, test_subject, is_outer, training_subject=None):
    """ Executes the training loop for the given test subject.

    Args:
        config (dict): The training configuration.
        test_subject (str): The test subject name.
    """
    print(colored(
        f"\n\n===========================================================\n" + 
        f"Rank {rank} is starting training for {test_subject} in {config['selected_model_name']}\n"
        , 'magenta'
    ))
    training_vars = training_preparation.TrainingVars(config, test_subject, is_outer, training_subject=training_subject)
    training_loop.training_loop(
        config, 
        test_subject, 
        training_vars.files, 
        training_vars.folds, 
        training_vars.n_folds, 
        training_vars.indexes, 
        training_vars.label_position,
        is_outer,
        rank
    )


def main(config_loc, is_outer):
    """ Runs the training process for each configuration and test subject. Process 0 DOES NOT train. """
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_proc = comm.Get_size()
    
    # Initalize TF
    tf_config = tf.compat.v1.ConfigProto()
    if rank == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        tf_config.gpu_options.visible_device_list = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank%2)
        tf_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=tf_config)
    
    # Rank 0 initializes the program and runs the configuration loops
    if rank == 0:  
         
        # Get start time
        start_time_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()
        
        # Get the configurations
        configs = parse_training_configs(config_loc)
        n_configs = len(configs)
        next_task_index = 0
        
        # No configs, no run
        if n_configs == 0:
            print(colored("No configurations given.", 'yellow'))
            for subrank in range(1, n_proc):
                comm.send(False, dest=subrank)
            exit()
        
        # Get the tasks for each process
        tasks = split_tasks(configs, n_proc, is_outer)
            
        # Listen for process messages while running
        exited = []
        while True:
            subrank = comm.recv(
                source=MPI.ANY_SOURCE
            )
            
            # Send task if the process is ready
            if tasks:
                print(colored(f"Rank 0 is sending rank {subrank} their first task.", 'green'))
                comm.send(
                    tasks.pop(), 
                    dest=subrank
                )
                next_task_index += 1
                    
            # If no task remains, terminate process
            else:
                print(colored(f"Rank 0 is terminating rank {subrank}, no tasks to give.", 'red'))
                comm.send(
                    False, 
                    dest=subrank
                )
                exited += [subrank]
                
                # Check if any processes are left, end this process if so
                if all(subrank in exited for subrank in range(1, n_proc)):
                    
                    # Get end time and print
                    print(colored(f"Rank 0 is printing the processing time.", 'red'))
                    elapsed_time = time.time() - start_time
                    with open(f'_TIME_MPI_{start_time_name}.txt', 'w') as fp:
                        fp.write(f"{elapsed_time}")
                    print(colored(f'Rank {rank} terminated. All other processes are finished.', 'yellow'))
                    break
            
    # The other ranks will listen for rank 0's messages and run the training loop
    else: 
        tf.config.run_functions_eagerly(True)
        
        # Listen for the first task
        print(colored(f'Rank {rank} is listening for process 0.', 'cyan'))
        comm.send(rank, dest=0)
        task = comm.recv(source=0)
        
        # While there are tasks to run, train
        while task:
                    
            # Training loop
            config, test_subject, training_subject = task
            run_training(rank, config, test_subject, training_subject, is_outer)
            comm.send(rank, dest=0)
            task = comm.recv(source=0)
            
        # Nothing more to run.
        print(colored(f'Rank {rank} terminated. All jobs finished for this process.', 'yellow'))
        

if __name__ == "__main__":
    """ Called when this file is run. """
    main()
