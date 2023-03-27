from training.training_modules.data_processing import training_preparation
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
            
def config_loop(rank, config, test_subjects, is_outer):
    
    # Read in the log's subject list, if it exists
    log_list = read_log_items(
        config['output_path'], 
        config['job_name'], 
        ['test_subjects'],
        rank
    )
    if log_list and 'subject_list' in log_list:
        test_subjects = log_list['test_subjects']
    
    # Check if no test subjects
    if not test_subjects:
        print(colored(f"Rank {rank} has no test subjects to loop through.", 'cyan'))
        return
    
    # Loop through all test subjects0
    print(colored(f"Rank {rank} is starting the configuration loop.", 'cyan'))
    for test_subject in test_subjects:
        subject_loop(rank, config, test_subject, is_outer)
        write_log(
            config['output_path'], 
            config['job_name'], 
            {'test_subjects': [t for t in test_subjects if t != test_subject]},
            rank
        )
        
        
def subject_loop(rank, config, test_subject, is_outer):
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
    training_vars = training_preparation.TrainingVars(config, test_subject, is_outer)
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
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        tf_config.gpu_options.visible_device_list = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(rank%2)
        tf_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=tf_config)
    
    # Equally spread GPU allocation
    #tf.debugging.set_log_device_placement(True)
    #strategy = tf.distribute.MirroredStrategy(tf.config.list_logical_devices('GPU'))
    
    # Rank 0 initializes the program and runs the configuration loops
    if rank == 0:   
        # Get start time
        start_time_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()
        
        # Print the environment info TODO move to rank 1?
        # console_printing.show_gpu_list()
        # console_printing.show_cpu_count()
        # console_printing.show_process_count(comm)
        
        # Get the configurations
        configs = parse_training_configs(config_loc)
        n_configs = len(configs)
        current_configs = {}
        
        # No configs, no run
        if n_configs == 0:
            print(colored("No configurations given.", 'yellow'))
            for subrank in range(1, n_proc):
                comm.send((None, None), dest=subrank, tag=1)
            exit()
        
        # Loop through them all and get the splits
        config_splits = {}
        for config_index, config in enumerate(configs):
            config_splits[config_index] = {}
            
            # Separate and store test subjects
            """n_split = math.ceil(len(config['test_subjects'])/(n_proc-1))
            for subrank in range(1, n_proc):
                config_splits[index][subrank] = config['test_subjects'][(subrank-1)*n_split:subrank*n_split]"""
            for subject_index, subject in enumerate(config['test_subjects']):
                subrank = (subject_index%(n_proc-1)) + 1
                if subrank not in config_splits[config_index]:
                    config_splits[config_index][subrank] = []
                config_splits[config_index][subrank] += [subject]
            
        # Communicate to each process for the first time
        for subrank in range(1, n_proc):
            print(colored(f"Rank 0 is sending rank {subrank} their first task.", 'green'))
            current_configs[subrank] = 0
            comm.send(
                (configs[0], config_splits[0][subrank]), 
                dest=subrank, 
                tag=1
            )
            
        # Loop while the subprocesses are running.
        exited = []
        while True:
            subrank, subrank_msg = comm.recv(
                source=MPI.ANY_SOURCE
            )
            
            # Check if a process sent a termination signal
            if subrank_msg is None:
                
                # Add to list of terminated processes
                exited.append(subrank)
                
                # Check if any processes are left, end this process if so
                if all(subrank in exited for subrank in range(1, n_proc)):
                    
                    # Get end time and print
                    print(colored(f"Rank 0 is printing the processing time.", 'red'))
                    elapsed_time = time.time() - start_time
                    with open(f'_TIME_MPI_{start_time_name}.txt', 'w') as fp:
                        fp.write(f"{elapsed_time}")
                    print(colored(f'Rank {rank} terminated. All other processes are finished.', 'yellow'))
                    break
            
            # Send next job if one is available
            current_configs[subrank] += 1
            if current_configs[subrank] < n_configs:
                print(colored(f"Rank 0 is sending rank {subrank} their next task.", 'green'))
                current_config = configs[current_configs[subrank]]
                comm.send(
                    (current_config, config_splits[current_configs[subrank]][subrank]), 
                    dest=subrank
                )
                
            # If nothing more to run, tell process to terminate
            else:
                print(colored(f"Rank 0 is terminating rank {subrank}.", 'red'))
                comm.send(
                    (None, None), 
                    dest=subrank
                )
                
            
    # The other ranks will listen for rank 0's messages and run the training loop
    else:        
        tf.config.run_functions_eagerly(True)
        
        # Listen for the config and test subject list
        print(colored(f'Rank {rank} is listening for process 0.', 'cyan'))
        comm.send((rank, rank), dest=0)
        config, test_subjects = comm.recv(source=0)
        
        # Run the items through the training algorithm
        while config:
            config_loop(rank, config, test_subjects, is_outer) 
            
            # Listen for the next job
            print(colored(f'Rank {rank} is listening for process 0.', 'cyan'))
            comm.send((rank, rank), dest=0)
            config, test_subjects = comm.recv(source=0)
            
        # Send termination
        comm.send((rank, None), dest=0)
            
        # Nothing more to run.
        print(colored(f'Rank {rank} terminated. All jobs finished for this process.', 'yellow'))
        

if __name__ == "__main__":
    """ Called when this file is run. """
    main()
