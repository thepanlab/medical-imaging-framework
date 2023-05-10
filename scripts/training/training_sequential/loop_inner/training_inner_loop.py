from training.training_sequential import sequential_processing
import datetime
import time
import os

# Location of the configurationss
CONFIG_LOC = './training/training_config_files/loop_inner'

# Sequential Inner Loop
if __name__ == "__main__":
    """ Called when this file is run. """   
    # Get the start time
    start_time_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()
    
    # Run job
    sequential_processing.main(
        config_loc=CONFIG_LOC, 
        is_outer=False
    )
    
    # Write elapsed time
    elapsed_time = time.time() - start_time
    if not os.path.exists("../results/training_timings"):
        os.makedirs("../results/training_timings")
    with open(os.path.join("../results/training_timings", f'_TIME_SEQ_INNER_{start_time_name}.txt'), 'w') as fp:
        fp.write(f"{elapsed_time}")
