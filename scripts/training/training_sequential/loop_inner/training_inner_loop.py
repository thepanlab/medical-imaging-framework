import os
import time
import datetime
from training.training_sequential import sequential_processing


# Location of the configurationss
CONFIG_LOC = './training/training_config_files/loop_inner'

# Sequential Inner Loop
if __name__ == "__main__":
    """ Called when this file is run. """   
    # Get the start time
    start_time_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()
    start_perf = time.perf_counter()
    
    # Run job
    sequential_processing.main(
        config_loc=CONFIG_LOC, 
        is_outer=False
    )
    
    # Write elapsed time
    elapsed_time = time.perf_counter() - start_perf
    
    timing_dir = "../results/training_timings"
    os.makedirs(timing_dir, exist_ok=True)

    timing_file = f"_TIME_SEQ_INNER_{start_time_name}.txt"
    timing_path = os.path.join(timing_dir, timing_file)

    with open(timing_path, 'w') as fp:
        fp.write(str(elapsed_time))
