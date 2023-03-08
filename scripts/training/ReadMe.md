# Running the Inner Training Loop

So you want to run the inner training loop, huh? That can be done in just ***one*** simple step:
1. Run the command " *make training_inner_loop* " from the *scripts* folder.
   * The program automatically uses the folder: " *scripts/training/training_config_files/* "
      * To use a particular json file, run: " *make training_inner_loop args='--file ./path/to/file.json'* "
      * To use a particular folder that contains one or more json files, run: " *make training_inner_loop args='--folder ./path/to/folder/'* "
   * All of the input file paths should be within the context of the *scripts* folder. For example: "*./training/custom_path/example.json*"

<br> <hr>

# Modifying Dependencies

Do you want *only* to have the inner training loop code, and not the processing output code? 
You can do this if you modify the "*_init_.py*" file within the "*scripts*" folder. Just remove the dependency you do not wish to have. 
Note that the training loop depends on *util* for configuration file parsing.

<br>

Or, do you want to add a new module? You will have to create and/or modify "*_init_.py*" files within the particular directory the module is located in. 
If this is not done, the program will not be able to find your particular module.


