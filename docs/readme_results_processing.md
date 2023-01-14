# **Results Processing**
<ul> 
    This module contains many various scripts that help process data from the data and training results. This can help to determine various performance metrics of the model. It does not directly affect the training process whatsoever. 
    All of these submodules can be run manually, or by using the provided Makefile. View the makefile or its documentation for more details.

</ul> <hr> <br> 

+ ## ***Confusion Matrix***
    <ul> 
        These scripts are responsible for creating confusion matrices for each test fold, validation fold, and configuration. If enough data is available, the <i>graphing</i> file will also produce the average and standard error matrices for each test fold and configuration pair.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***confusion_matrix.py:***
        ***Example:*** 
        >python3 confusion_matrix.py -j my_config.json

        <details>
        
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** A confusion matrix in CSV format.
        * ***confusion_matrix_config.json:***
            ```json
                {
                    "pred_path": "[path]/example_test_X_val_Y_val_predicted_index.csv",
                    "true_path": "[path]/example_test_X_val_Y_val true label index.csv",

                    "output_path": "[path]/confusion_matrix",
                    "output_file_prefix": "example_test_X_val_Y_val",

                    "label_types": [ "A", "B", "C" ]
                }
            ```
            * ***pred_path:*** The file path to the *indexed* predicted values of a specific testing fold, validation fold, and configuration.
            * ***true_path:*** The file path to the indexed true values of a specific testing fold, validation fold, and configuration.
            * ***output_path:*** The directory path to where the CSV file should be written. 
            * ***output_file_prefix:*** This will result in a file named "*[prefix]_conf_matrix.csv*" 
            * ***label_types:*** These are the labels that will appear on the output matrix.

        </details> <hr> <br>

    
    2) ### ***confusion_matrix_many.py:***
        ***Example:*** 
        >python3 confusion_matrix_many.py -j my_config.json

        <details>

        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** Many confusion matrices in CSV format.
        * ***confusion_matrix_many_config.json:***
            ```json
                {
                    "data_path": "[path]/data",
                    "output_path": "[path]/confusion_matrix",

                    "label_types": { "A": 0, "B": 1, "C": 2 }
                }
            ```
            * ***data_path:*** The file path to the overall data. This folder should contain the testing folds' directories.
            * ***output_path:*** The directory path to where the confusion matrix CSV files should be written.
            * ***label_types:*** These are the labels that will appear on the output matrix.

        </details> <hr> <br>
    
    3) ### ***confusion_matrix_many_means.py:***
        ***Example:*** 
        >python3 confusion_matrix_many_means.py -j my_config.json

        <details>

        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** Many confusion means in CSV format and their standard error.
        * ***confusion_matrix_many_means_config.json:***
            ```json
                {
                    "data_path": "[path]/data",
                    "matrices_path": "[path]/confusion_matrix",
                    "means_output_path": "[path]/confusion_matrix_means",
        
                    "round_to": 2,

                    "label_types": [ "A", "B", "C" ]
                }
            ```
            * ***data_path:*** The file path to the overall data. This folder should contain the testing folds' directories.
            * ***matrices_path:*** The directory path to the confusion matrices.
            * ***means_output_path:*** The directory path to where the confusion matrix CSV files should be written.
            * ***round_to:*** Allows for the rounding of output values.
            * ***label_types:*** These are the labels that will appear on the output matrix.
        
        </details> <hr> <br> <br> 
    </details>
<hr>


+ ## ***Epoch Counting***
    <ul> 
        This script will count the number of epochs that occured in the early stopping phase of each validation fold, testing fold, and configuration. It will also produce the average and standard error for every testing fold and configuration.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***epoch_counting.py:***
        ***Example:*** 
        >python3 epoch_counting.py -j my_config.json

        <details>

        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** Two CSV files. One for the every epoch count. The other for the average and standard error of each testing fold.
        * ***epoch_counting_config.json:***
            ```json
                {
                    "data_path": "[path]/data/",
                    "output_path": "[path]/epoch_output/"
                }
            ```
            * ***data_path:*** The file path to the overall data. This folder should contain the testing folds' directories.
            * ***output_path:*** The directory path to where the confusion matrix CSV files should be written.

        <details> <hr> <br> <br>
    </details>     
<hr>



+ ## ***Grad Cam***
    <ul> 
        This script takes in an image and draws the most influential areas of the configuration by using a heatmap. 
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***grad_cam.py:***
        ***Example:*** 
        >python3 grad_cam.py -j my_config.json

        <details>

        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** One or more images.
        * ***grad_cam_config.json:***
            ```json
                {
                    "input_model_address": "[path]/model.h5",
                    "input_img_address": "[path]/img.png",
                    "output_image_address": "[path]/grad_cam",
                    "alpha": 0.7,
                    "last_conv_layer_name": ""
                }
            ```
            * ***input_model_address:*** The filepath to the trained model.
            * ***input_img_address:*** The input image to alter.
            * ***output_image_address:*** The address to output an image to.
            * ***alpha:*** The new image's alpha value.
            * ***last_conv_layer_name:*** The particular layer name within the model. If not provided, is automatically detected. *(Optional)*

        <details> <hr> <br>

    2) ### ***grad_cam_many.py:***
        ***Example:*** 
        >python3 grad_cam_many.py -j my_config.json

        <details>

        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** One or more images.
        * ***grad_cam_many.json:***
            ```json
                {
                    "input_model_address": "[path]/model.h5",
                    "input_img_address": "[path]/img.png",
                    "output_image_address": "[path]/grad_cam",
                    "alpha": 0.7,
                    "last_conv_layer_name": "",

                    "query":{
                        "cutoff_number_of_results": -1,
                        "sort_images": true,

                        "test_subject": ["k2", "k4"],

                        "match": [true, false],

                        "true_label": ["ligament"],
                        "true_label_index": [],

                        "pred_label": [],
                        "pred_label_index": [2, 3],

                        "true_predicted_label_pairs": [],
                        "true_predicted_index_pairs": [[1, 2], [3, 3]] 
                }
            ```
            * ***input_model_address:*** The filepath to the trained model.
            * ***input_img_address:*** The input image to alter.
            * ***output_image_address:*** The address to output an image to.
            * ***alpha:*** The new image's alpha value.
            * ***last_conv_layer_name:*** The particular layer name within the model. If not provided, is automatically detected. *(Optional)*
            * ***query:*** All of the arguements within the query are entirely optional. This chooses specific images given the input path. For example, setting *match* to true will only output images with correct predictions. *(Optional)*

        </details> <br> <br>
    </details>
<hr>


+ ## ***Learning Curve***
    <ul> These scripts will take the loss and accuracy values from each iteration to produce two graphs. This will happen for every testing fold, validation fold, and configuration.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***learning_curve.py:***
        ***Example:*** 
        >python3 learning_curve.py -j my_config.json

        <details>

        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** Two images. One for accuracy and the other for loss.
        * ***learning_curve_config.json:***
            ```json
                {
                    "input_path": "[path]/[config]/[test fold]/",
                    "output_path": "[path]/learning_curve",

                    "loss_line_color": "r",
                    "val_loss_line_color": "b",
                    "acc_line_color": "b",
                    "val_acc_line_color": "r",

                    "font_family": "DejaVu Sans",
                    "label_font_size": 12,
                    "title_font_size": 12,

                    "save_resolution": 600,
                    "save_format": "png"
                }
            ```
            * ***input_path:*** The directory path of a particular testing fold.
            * ***output_path:*** The directory path to where the PNG files should be written. 
            * ***loss_line_color:*** Color of the loss line. 
            * ***val_loss_line_color:*** Color of the validation loss line.
            * ***acc_line_color:*** Color of the accuracy line.
            * ***val_acc_line_color:*** Color of the validation accuracy line. 
            * ***font_family:*** The font to be used with the PyPlot graphing tool.
            * ***label_font_size:*** Size of the axis label fonts.
            * ***title_font_size:*** Size of the title font. 
            * ***save_resolution:*** Resolution of the image output.
            * ***save_format:*** Image type to save.

        </details> <hr> <br>
    

    2) ### ***learning_curve_many.py:***
        ***Example:*** 
        >python3 learning_curve_many.py -j my_config.json

        <details>

        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** Many images, for every testing fold, validation fold, and configuration.
        * ***learning_curve_many_config.json:***
            ```json
                {
                    "data_path": "[path]/data/",
                    "output_path": "[path]/learning_curve",

                    "loss_line_color": "r",
                    "val_loss_line_color": "b",
                    "acc_line_color": "b",
                    "val_acc_line_color": "r",

                    "font_family": "DejaVu Sans",
                    "label_font_size": 12,
                    "title_font_size": 12,

                    "save_resolution": 600,
                    "save_format": "png"
                }
            ```
            * ***data_path:*** The directory path of the data as a whole. This folder should contain the testing fold directories.
            * ***output_path:*** The directory path to where the PNG files should be written. 
            * ***loss_line_color:*** Color of the loss line. 
            * ***val_loss_line_color:*** Color of the validation loss line.
            * ***acc_line_color:*** Color of the accuracy line.
            * ***val_acc_line_color:*** Color of the validation accuracy line.
            * ***font_family:*** The font to be used with the PyPlot graphing tool.
            * ***label_font_size:*** Size of the axis label fonts.
            * ***title_font_size:*** Size of the title font. 
            * ***save_resolution:*** Resolution of the image output.
            * ***save_format:*** Image type to save.

        </details> <hr> <br> <br>
    </details>
<hr>



+ ## ***Metrics Table***
    <ul> This will create tables for each configuration, giving the accuracies and error of every test and validation fold pair.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***metrics_table.py:***
        ***Example:*** 
        > python3 metrics_table.py -j my_config.json

        <details>
        
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** A CSV file.
        * ***metrics_table_config.json:***
            ```json
                {
                    "data_path": "[path]/data/",
                    "output_path": "[path]/metrics_output",
                    "output_filename": "metrics_table",

                    "round_to": 6
                }
            ```
            * ***data_path:*** The directory path of the data as a whole. This folder should contain the testing fold directories.
            * ***output_path:*** The directory path to where the CSV file should be written. 
            * ***output_filename:*** This will result in a file named "*[name].csv*" 
            * ***round_to:*** This will allow the rounding of output values.

        </details> <hr> <br> <br>
    </details>
<hr>



+ ## ***Prediction***
    <ul> This will make a set of predictions, given some raw data.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***prediction.py:***
        ***Example:*** 
        > python3 prediction.py -j my_config.json

        <details>
        
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** A CSV file.
        *  ***prediction_config.json:***
           ```json {
                "prediction_output": "[path]/predictions",
                "test_subject_data_input": {
                    "subject_name": "[path]/Test_subject_[subject_name]"
                },
                "model_input": {
                    "model_name": "[path]/[model].h5"
                },
        
                "batch_size": 8,
                "output_tabled_info": true,
                "use_true_labels": false,
                    
                "image_settings": {
                    "mean": 0,
                    "use_mean": "false",
                    "class_names": "A,B,C,D",
                    "channels": 1,
                    "do_cropping" : "false",
                    "offset_height": 0,
                    "offset_width": 0,
                    "target_height": 241,
                    "target_width": 181
               }
            }
           ```
           * ***prediction_output:*** Where to output predictions.
           * ***test_subject_data_input:*** A dictionary of subject - path form.
           * ***model_input:*** A dictionary of model - path form. 
           * ***batch_size:*** The size of batches to predict with.
           * ***output_tabled_info:*** If true, will output tabled information of the predictions automatically.
           * ***use_true_labels:*** If false, the predictions will be made without any true labels. I.e. it will only guess what labels go to what image.
           * ***image_settings:*** How to alter the given images. Mainly should worry about 'class_names'.

        </details> <hr> <br> <br>
    </details>
<hr>



+ ## ***Roc Curve***
    <ul> These scripts will use the predicted and true values to produce a ROC (Receiver Operating Characteristic) curve. The output will yield an image.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***roc_curve.py:***
        ***Example:*** 
        >python3 roc_curve.py -j my_config.json

        <details>

        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** One image.
        * ***roc_curve_config.json:***
            ```json
                {
                    "pred_path": "[path]/example_test_X_val_Y_val_predicted.csv",
                    "true_path": "[path]/example_test_X_val_Y_val true label index.csv",
                    "output_path": "[path]/roc_curve",
                    "output_file_prefix": "roc_curve_example",

                    "line_width": "2",
                    "label_types": ["A", "B", "C"],
                    "line_colors": ["red", "blue", "yellow"],

                    "font_family": "DejaVu Sans",
                    "label_font_size": 12,
                    "title_font_size": 12,

                    "save_resolution": "figure",
                    "save_format": "png"
                }
            ```
            * ***pred_path:*** The file path to the *non-indexed* predicted values of a specific testing fold, validation fold, and configuration.
            * ***true_path:*** The file path to the indexed true values of a specific testing fold, validation fold, and configuration. 
            * ***output_path:*** The directory path to where the CSV file should be written. 
            * ***output_file_prefix:*** This will result in a file named "*[prefix]_conf_matrix.csv*" 
            * ***line_width:*** Width of the line. 
            * ***label_types:*** Axis labels.
            * ***line_colors:*** Color of the ROC curve.
            * ***font_family:*** The font to be used with the PyPlot graphing tool.
            * ***label_font_size:*** Size of the axis label fonts.
            * ***title_font_size:*** Size of the title font. 
            * ***save_resolution:*** Resolution of the image output.
            * ***save_format:*** Image type to save.
        
        </details> <hr> <br>
    
    2) ### ***roc_curve_many.py:***
        ***Example:*** 
        >python3 roc_curve_many.py -j my_config.json
        
        <details>

        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** Many images, for every testing fold, validation fold, and configuration.
          * ***roc_curve_many_config.json:***
          ```json
              {
                  "data_path": "[path]/data/",
                  "output_path": "[path]/roc_output",

                  "line_width": "2",
                  "label_types": ["A", "B", "C"],
                  "line_colors": ["red", "blue", "yellow"],

                  "font_family": "DejaVu Sans",
                  "label_font_size": 12,
                  "title_font_size": 12,

                  "save_resolution": "figure",
                  "save_format": "png"
              }
          ```
          * ***data_path:*** The directory path of the data as a whole. This folder should contain the testing fold directories.
          * ***output_path:*** The directory path to where the PNG files should be written. 
          * ***line_width:*** Width of the line. 
          * ***label_types:*** Axis labels.
          * ***line_colors:*** Color of the ROC curve.
          * ***font_family:*** The font to be used with the PyPlot graphing tool.
          * ***label_font_size:*** Size of the axis label fonts.
          * ***title_font_size:*** Size of the title font. 
          * ***save_resolution:*** Resolution of the image output.
          * ***save_format:*** Image type to save.

        </details> <hr> <br> <br>
    </details>
<hr>



+ ## ***Summary Table***
    <ul> This script will create a summary for each testing fold. It will contain the mean accuracy and standard error for each fold's configuration.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***summary_table.py:***
        ***Example:*** 
        > python3 summary_table.py -j my_config.json

        <details>

        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** A CSV file.
        * ***summary_table_config.json:***
            ```json
                {
                    "data_path": "[path]/data/",
                    "output_path": "[path]/summary_output",
                    "output_filename": "summary_table",

                    "round_to": 6
                }
            ```
            * ***data_path:*** The directory path of the data as a whole. This folder should contain the testing fold directories.
            * ***output_path:*** The directory path to where the CSV file should be written. 
            * ***output_filename:*** This will result in a file named "*[name].csv*" 
            * ***round_to:*** This will allow the rounding of output values.

       </details> <hr>  <br> <br>
    </details>
<hr>



+ ## ***Tabled Prediction Info***
    <ul> This will create a more detailed table of prediction information.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***tabled_prediction_info.py:***
        ***Example:*** 
        > python3 tabled_prediction_info.py -j my_config.json

        <details>

        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** A CSV file.
        * ***tabled_prediction_info.json:***
        ***Example:*** 
            ```json
                {
                    "data_path": "[path]/data/",
                    "output_path": "[path]/summary_output",
  
                    "use_true_labels": true,

                    "label_types": {"0":"A", "1":"B", "2":"C"}
                }
            ```
            * ***data_path:*** The directory path of the data as a whole. This folder should contain the testing fold directories.
            * ***output_path:*** The directory path to where the CSV file should be written.
            * ***use_true_labels:*** Specifies whether to look for an image's true label or not.
            * ***label_types:*** The labels and their index.

       </details> <hr> <br> <br>
    </details>
<hr>
