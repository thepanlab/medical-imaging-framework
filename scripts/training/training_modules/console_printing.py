from termcolor import colored
import tensorflow as tf


def show_gpu_list():
    """ This will print a list of the available GPUs on the current system. """
    gpus = tf.config.list_physical_devices('GPU')
    print(colored(f"\n-----------------------------------------", 'magenta'))
    print(colored("GPU list:", 'magenta'))
    for gpu in gpus:
        print(colored(f"Name: {gpu.name}  | Type: {gpu.device_type}", 'cyan'))
