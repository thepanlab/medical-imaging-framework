# **Makefile**

<ul> 
    The makefile has the ability to run every runnable module within this package. The command lists are given here.
</ul> <hr> <br> 

+ ## ***Results Processing***
    All of the result processing modules share the same makefile command line format.
    > make [MODULE NAME] -j [JSON PATH]
    
    * **MODULE NAME** is the name of the module. This is also the script's name, but without the *.py* extention.
    * **JSON PATH** Is the path to the input configuration file. If blank, you will be prompted for one. Entering nothing will make the program use the default file location. *(Optional)*

    <br>

    The possible result processing modules to call are as follows:
    * *class_accuracy*
    * *confusion_matrix*
    * *confusion_matrix_many*
    * *confusion_matrix_many_means*
    * *epoch_counting*
    * *grad_cam*
    * *grad_cam_many*
    * *learning_curve*
    * *learning_curve_many*
    * *metrics_table*
    * *prediction*
    * *roc_curve*
    * *roc_curve_many*
    * *roc_curve_many_means*
    * *summary_table*
    * *tabled_prediction_info*

<hr> <br> <br>


+ ## ***Training***
    
    **Modules:**
    * *training_inner_loop*
    * *training_outer_loop*
    * *distributed_training_inner_loop*
    * *distributed_training_outer_loop*
    * *mpi_init*
    * *predicted_formatter*
    * *truth_formatter*

    <br>

    The *mpi_init* module generates a command line arguement to copy-paste if the makefile does not suffice.
    The other makefile modules have a specific format. For example:
    > make training_inner_loop

    or
    > make training_inner_loop args='--file FILE PATH'

    or
    > make training_inner_loop args='--folder FOLDER PATH'
    
    * **FILE PATH** Is the path to a specific configuration file. *(Optional)*
    * **JSON PATH** Is the path to a specific configuration folder of configuration file. *(Optional)*

<hr> <br> <br>

