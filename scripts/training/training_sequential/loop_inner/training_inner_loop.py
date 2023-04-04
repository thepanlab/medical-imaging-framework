from training import run_training
import datetime
import time

# Location of the configurationss
CONFIG_LOC = './training/training_config_files/loop_inner'

if __name__ == "__main__":
    """ Called when this file is run. """   
    
    start_time_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()
    run_training.main(
        config_loc=CONFIG_LOC, 
        is_outer=False
    )
    elapsed_time = time.time() - start_time
    with open(f'_TIME_SEQ_{start_time_name}.txt', 'w') as fp:
        fp.write(f"{elapsed_time}")


