from util.get_config import parse_json
from termcolor import colored
import subprocess
import socket
import sys
import os


def main():
    """ The main program. """
    # Read in the program configuration
    config = parse_json("./multiprocessed_training/mpi_config.json")
    
    # We will create a command line argument
    arg = ""
    
    # If the devices are specified, add them to the beginning
    if config['cuda_devices']:
        devices = ','.join(map(str, config['cuda_devices']))
        print(colored(f"Specified CUDA devices: {devices}", 'cyan'))
        arg += f"CUDA_VISIBLE_DEVICES={devices} "
    
    # Get the worker (process) addresses, if they exist
    if config['gpu_addrs']:
        print(colored(f"Specified IP addresses: {','.join(config['gpu_addrs'])}", 'cyan'))
        
        # Try to make sure every address is reachable
        for addr in config['gpu_addrs']:
            print(colored(f"\nAttempting to ping {addr}...", 'blue'))
            if os.system("ping -c 1 " + addr) != 0:
                raise Exception(colored(f'Error: Host {addr} could not be reached.', 'red'))
            
        # One process is added to the host machine for the master (non-training) process
        ip_address = socket.gethostbyname(socket.gethostname())
        addrs = [ip_address] + config['gpu_addrs']
        arg += f"mpirun -H {','.join(addrs)} "
        
    # If not, just run n processes on the same machine
    else:
        print(colored(f"Specified processes: {config['n_processes']}", 'cyan'))
        
        # One process is added to the host machine for the master (non-training) process
        arg += f"mpirun -n {int(config['n_processes']) + 1} " 
        
    # Add the final location of the program
    arg += "python3 -m multiprocessed_training.mpi_processing "
    print(colored(f"\nRunning argument: {arg}", 'green'))
    print(colored(f"Note: An extra host CPU process is added automatically to what is given.", 'cyan'))
    
    # Run the command line arguement
    """
    sp = subprocess.Popen(
        arg, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True
    )
    while sp.poll() is None:
        print(sp.stdout.readline())
    """
    

if __name__ == '__main__':
    """ Executes the main function """
    main()
