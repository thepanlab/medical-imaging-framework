from termcolor import colored
import tensorflow as tf
import multiprocessing
from mpi4py import MPI


def show_gpu_list():
    """ This will print a list of the available GPUs on the current system. """
    gpus = tf.config.list_physical_devices('GPU')
    print(colored(f"\n=========================================", 'magenta'))
    print(colored("GPU list:", 'magenta'))
    if not gpus:
        print(colored("None", 'cyan'))
    else:
        for gpu in gpus:
            print(colored(f"Name: {gpu.name}  | Type: {gpu.device_type}", 'cyan'))
    return gpus


def show_cpu_count():
    """ This will print the number of available CPUS. """
    cpus = multiprocessing.cpu_count()
    print(colored(f"-----------------------------------------", 'magenta'))
    print(colored("CPU count:", 'magenta'))
    print(colored(cpus, 'cyan'))


def show_process_count(comm):
    """ This will print the number of processes specified by the user. """
    procs = comm.Get_size()
    print(colored(f"-----------------------------------------", 'magenta'))
    print(colored("Process count:", 'magenta'))
    print(colored(procs, 'cyan'))
    print(colored(f"=========================================", 'magenta'))


