# **Training**

<ul> 
    This module contains many various scripts that help process data from the data and training results. This can help to determine various performance metrics of the model. It does not directly affect the training process whatsoever. 
    All of these submodules can be run manually, or by using the provided Makefile. View the makefile or its documentation for more details.

</ul> <hr> <br> 

+ ## ***training_inner_loop.py***
    <ul> 
        This is the main file of the entire training process. This can be called using the Makefile and manually within the command line from within the scripts folder. Do to its modularization implementation, it will most likely break when called outside of this location. The default configuration folder is "training/training_config_files".
    </ul>

    ***Example:*** 
    >python3 -m training.training_inner_loop --file myfile.json OR --folder /myfolder

    <details>

    * ***Input:*** The configuration file or folder. *(Optional)*
    * ***Output:*** Trained models, prediction results, and other various metrics.
    * ***example_model_config.json:***
        ```json
        {
            "hyperparameters": {
                "batch_size": 24,
                "channels": 1,
                "cropping_position": [40, 10],
                "decay": 0.01,
                "do_cropping": true,
                "epochs": 6,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "patience": 10
            },

            "data_input_directory": "[path]/ep_intensity",
            "output_path": "[path]/training",
            "job_name": "InceptionV3_test_1",
            
            "checkpointing_title": "checkpointing_test",
            "k_epoch_checkpoint_frequency": 2,

            "shuffle_the_folds": true,
            "n_folds": "all",
            "seed": 9,

            "class_names": ["fat", "ligament", "flavum", "epidural space", "spinal cord"],
            "selected_model_name": "InceptionV3",
            "subject_list": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "test_subjects": ["e1", "e2"],

            "image_size": [800, 400],
            "target_height": 241,
            "target_width": 181
        }

        ```
        * ***hyperparameters:*** These are the parameters needed for the training step.
          * ***batch_size:*** This will divide the input datasets into n training batches.
          * ***channels:*** The image channels in which the image will be processed on.
          * ***cropping_position:*** The position at which to crop the image.
          * ***decay:*** Decays the learning rate over time.
          * ***do_cropping:*** Whether to crop the input images.
          * ***epochs:*** The number of training epochs.
          * ***learning_rate:*** The learning speed.
          * ***momentum:*** Helps the learning rate's speed by speeding up the gradient descent search.
          * ***patience:*** How long to wait for improvement within early stopping.
        * ***data_input_directory:*** Where the input images are located. No specific structure is needed.
        * ***output_path:*** Where to write the results to.
        * ***job_name:*** The name of your job. Will mainly affect checkpointing file names.
        * ***checkpointing_title:*** The title of your checkpoint files.
        * ***k_epoch_checkpoint_frequency:*** How many epochs should checkpoints be saved.
        * ***shuffle_the_folds:*** Whether to randomly shuffle the training folds.
        * ***n_folds:*** How many folds to include. Use either an int or "all".
        * ***seed:*** The random seed.
        * ***class_names:*** The names of the image classes or labels.
        * ***selected_model_name:*** The model type to create. The choices are: resnet_50, resnet_VGG16, InceptionV3, ResNet50V2, and Xception.
        * ***subject_list:*** The list of all training subjects. 
        * ***test_subjects:*** The list of the particular testing subjects.
        * ***image_size:*** The expected image size.
        * ***target_height:*** The target image height.
        * ***target_width:*** The target image width.
  
    </details> </hr> <br> <br>
<hr>



+ ## ***Checkpoint Modules***
    <ul> 
        This module contains various scripts that aid in checkpointing the model and continuing from previous training data.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***checkpointer.py:***
        <ul> 
            This has the ability to write and load checkpoints. This allows the model to continue off from a previous training session. The checkpoints are saved in the same format as regular models, in h5 form.
        </ul>

    2) ### ***logger.py:***
        <ul> 
            This allows the training loop to log the most recent training state. This will allow the training loop to carry off from a cancelled job. Things like the testing subject, validation subject, and various training fold properties are stored using the functions within. It has the ability to read, write, and delete log files.
        </ul>

    </details> <br> <br>
<hr>



+ ## ***Training Modules***
    <ul> 
        This module contains most of the main training loop functions.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***console_printing.py:***
        <ul> 
            This contains a basic printing function. It may be removed later.
        </ul>

    2) ### ***fold_generator.py:***
        <ul> 
            This will generate a list of training folds and the number of rotations for a given subject.
        </ul>

    3) ### ***image_getter.py:***
        <ul> 
            This retrieves every image from the given input file path.
        </ul>

    4) ### ***image_parser.py:***
        <ul> 
            This parses the input images as tensors.
        </ul>

    5) ### ***index_getter.py:***
        <ul> 
            This will generate ordered lists for the labels, label-indexes, and subjects of the input images from their names.
        </ul>

    6) ### ***model_creator.py:***
        <ul> 
            This generates a model object, based on the given configuration.
        </ul>

    7) ### ***result_outputter.py:***
        <ul> 
            This outputs various training metrics after the process is done within each fold.
        </ul>

    8) ### ***training_fold.py:***
        <ul> 
            This is the main training function. Here is where the model is trained and its data is saved within a log and checkpoint.
        </ul>

    9) ### ***training_loop.py:***
        <ul> 
            This module runs all of the training folds for a particular subject.
        </ul>

    10) ### ***training_preparation.py:***
        <ul> 
            This is where basic training loop data is generated. Things like files, indexes, and folds are created here.
        </ul>

    </details> <br> <br>
<hr>


