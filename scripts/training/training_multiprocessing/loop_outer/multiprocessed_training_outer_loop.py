# Location of the configurations
CONFIG_LOC = './training/training_config_files/loop_outer'

from training.training_multiprocessing import mpi_processing
if __name__ == "__main__":
    """ Called when this file is run. """   
    mpi_processing.main(
        config_loc=CONFIG_LOC, 
        is_outer=True
    )
