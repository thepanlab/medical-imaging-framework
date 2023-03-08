from training.training_multiprocessing import mpi_processing

# Location of the configurationss
CONFIG_LOC = './training/training_config_files/loop_outer'

if __name__ == "__main__":
    """ Called when this file is run. """   
    mpi_processing.main(
        config_loc=CONFIG_LOC, 
        is_outer=True
    )
