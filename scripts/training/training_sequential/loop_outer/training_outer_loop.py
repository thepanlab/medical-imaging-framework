from training import run_training

# Location of the configurationss
CONFIG_LOC = './training/training_config_files/loop_outer'

if __name__ == "__main__":
    """ Called when this file is run. """   
    run_training.main(
        config_loc=CONFIG_LOC, 
        is_outer=True
    )
