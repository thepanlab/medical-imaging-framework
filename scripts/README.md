# **Scripts**

+ ## ***Confusion Matrix***
    <ul> 
        These scripts are responsible for creating confusion matricies for each test fold, validation fold, and configuration. If enough data is available, the <i>graphing</i> file will also produce the average and standard error matrices for each test fold and configuration pair.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***confusion_matrix.py:***
        ***Example:*** 
        >python3 confusion_matrix.py -j my_config.json
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** A confusion matrix in CSV format.
        
        <br>
    
    2) ### ***confusion_matrix_config.json:***
        ***Example:*** 
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
        
        <br>

        * ***output_path:*** The directory path to where the CSV file should be written. 
        * ***output_file_prefix:*** This will result in a file named "*[prefix]_conf_matrix.csv*" 
        
        <br>

        * ***label_types:*** These are the labels that will appear on the output matrix.

        <br>
    
    3) ### ***confusion_matrix_graphing.py:***
        ***Example:*** 
        >python3 confusion_matrix_graphing.py -j my_config.json
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** Many confusion matrices in CSV format, as well as their average and standard error counterparts.
        
        <br>
    
    2) ### ***confusion_matrix_graphing_config.json:***
        ***Example:*** 
        ```json
            {
                "data_path": "[path]/data",
                "output_path": "[path]/confusion_matrix",
                "average_stderr_output_path": "[path]/confusion_matrix_avg_stderr",

                "label_types": [ "A", "B", "C" ]
            }
        ```
        * ***data_path:*** The file path to the overall data. This folder should contain the testing folds' directories.
        * ***output_path:*** The directory path to where the confusion matrix CSV files should be written.
        * ***confusion_matrix_avg_stderr:*** The directory path to where the confusion matrix average and standard error CSV files should be written. 

        <br>

        * ***label_types:*** These are the labels that will appear on the output matrix.
        
        <br> <br> 
    </details>
        


- - -
+ ## ***Epoch Counting***
    <ul> 
        This script will count the number of epochs that occured in the early stopping phase of each validation fold, testing fold, and configuration. It will also produce the average and standard error for every testing fold and configuration.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***epoch_counting.py:***
        ***Example:*** 
        >python3 epoch_counting.py -j my_config.json
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** Two CSV files. One for the every epoch count. The other for the average and standard error of each testing fold.
        
        <br>
    
    2) ### ***epoch_counting_config.json:***
        ***Example:*** 
        ```json
            {
                "data_path": "[path]/data/",
                "output_path": "[path]/epoch_output/"
            }
        ```
        * ***data_path:*** The file path to the overall data. This folder should contain the testing folds' directories.
        * ***output_path:*** The directory path to where the confusion matrix CSV files should be written.

        <br> <br>
    </details>
        


- - -
+ ## ***Grad Cam***
    <ul> 
        This script takes in an image and draws the most influential areas of the configuration by using a heatmap. 
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***grad_cam.py:***
        ***Example:*** 
        >python3 grad_cam.py -j my_config.json
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** One or more images.
        
        <br>
    
    2) ### ***grad_cam_config.json:***
        ***Example:*** 
        ```json
            {
                "pred_path": "[path]/example_test_X_val_Y_val_predicted_index.csv",
                "true_path": "[path]/example_test_X_val_Y_val true label index.csv",

                "output_path": "[path]/confusion_matrix",
                "output_file_prefix": "example_test_X_val_Y_val",

                "label_types": [ "A", "B", "C" ]
            }
        ```
        * ***pred_path:*** The file path to the indexed predicted values of a specific testing fold, validation fold, and configuration.
        * ***true_path:*** The file path to the indexed true values of a specific testing fold, validation fold, and configuration. 
        
        <br>

        * ***output_path:*** The directory path to where the CSV file should be written. 
        * ***output_file_prefix:*** This will result in a file named "*[prefix]_conf_matrix.csv*" 
        
        <br>

        * ***label_types:*** These are the labels that will appear on the output matrix. 

        <br> <br>
    </details>
        


- - -
+ ## ***Learning Curve***
    <ul> These scripts will take the loss and accuracy values from each iteration to produce two graphs. This will happen for every testing fold, validation fold, and configuration.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***learning_curve.py:***
        ***Example:*** 
        >python3 learning_curve.py -j my_config.json
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** Two images. One for accuracy and the other for loss.
        
        <br>
    
    2) ### ***learning_curve_config.json:***
        ***Example:*** 
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

        <br>

        * ***loss_line_color:*** Color of the loss line. 
        * ***val_loss_line_color:*** Color of the validation loss line.
        * ***acc_line_color:*** Color of the accuracy line.
        * ***val_acc_line_color:*** Color of the validation accuracy line. 
        
        <br>

        * ***font_family:*** The font to be used with the PyPlot graphing tool.
        * ***label_font_size:*** Size of the axis label fonts.
        * ***title_font_size:*** Size of the title font. 
        
        <br>

        * ***save_resolution:*** Resolution of the image output.
        * ***save_format:*** Image type to save.

        <br>
    
    3) ### ***learning_curve_graphing.py:***
        ***Example:*** 
        >python3 learning_curve_graphing.py -j my_config.json
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** Many images, for every testing fold, validation fold, and configuration.
        
        <br>
    
    2) ### ***learning_curve_graphing_config.json:***
        ***Example:*** 
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

        <br>

        * ***loss_line_color:*** Color of the loss line. 
        * ***val_loss_line_color:*** Color of the validation loss line.
        * ***acc_line_color:*** Color of the accuracy line.
        * ***val_acc_line_color:*** Color of the validation accuracy line. 
        
        <br>

        * ***font_family:*** The font to be used with the PyPlot graphing tool.
        * ***label_font_size:*** Size of the axis label fonts.
        * ***title_font_size:*** Size of the title font. 
        
        <br>

        * ***save_resolution:*** Resolution of the image output.
        * ***save_format:*** Image type to save.

        <br> <br>
    </details>
        


- - -
+ ## ***Roc Curve***
    <ul> These scripts will use the predicted and true values to produce a ROC (Receiver Operating Characteristic) curve. The output will yield an image.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***roc_curve.py:***
        ***Example:*** 
        >python3 roc_curve.py -j my_config.json
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** One image.
        
        <br>
    
    2) ### ***roc_curve_config.json:***
        ***Example:*** 
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

        <br>

        * ***line_width:*** Width of the line. 
        * ***label_types:*** Axis labels.
        * ***line_colors:*** Color of the ROC curve.
        
        <br>

        * ***font_family:*** The font to be used with the PyPlot graphing tool.
        * ***label_font_size:*** Size of the axis label fonts.
        * ***title_font_size:*** Size of the title font. 
        
        <br>

        * ***save_resolution:*** Resolution of the image output.
        * ***save_format:*** Image type to save.

        <br>
    
    3) ### ***roc_curve_graphing.py:***
        ***Example:*** 
        >python3 roc_curve_graphing.py -j my_config.json
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** Many images, for every testing fold, validation fold, and configuration.
        
        <br>
    
    2) ### ***roc_curve_graphing_config.json:***
        ***Example:*** 
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

        <br>

        * ***line_width:*** Width of the line. 
        * ***label_types:*** Axis labels.
        * ***line_colors:*** Color of the ROC curve.
        
        <br>

        * ***font_family:*** The font to be used with the PyPlot graphing tool.
        * ***label_font_size:*** Size of the axis label fonts.
        * ***title_font_size:*** Size of the title font. 
        
        <br>

        * ***save_resolution:*** Resolution of the image output.
        * ***save_format:*** Image type to save.

        <br> <br>
    </details>
        


- - -
+ ## ***Summary Table***
    <ul> This script will create a summary for each testing fold. It will contain the mean accuracy and standard error for each fold's configuration.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***summary_table.py:***
        ***Example:*** 
        > python3 summary_table.py -j my_config.json
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** A CSV file.
        
        <br>
    
    2) ### ***summary_table_config.json:***
        ***Example:*** 
        ```json
            {
                "data_path": "[path]/data/",
                "output_path": "[path]/summary_output",
                "output_filename": "summary_table"
            }
        ```
        * ***data_path:*** The directory path of the data as a whole. This folder should contain the testing fold directories.
        * ***output_path:*** The directory path to where the CSV file should be written. 
        * ***output_filename:*** This will result in a file named "*[name].csv*" 

        <br> <br>
    <details>
    <summary>Summary Table Files</summary>
        


- - -
+ ## ***Others***
    <ul> There are other scripts included that are dependencies for the above items. They are not intended for direct use.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***get_config.py:***
        This will take in a configuration file path and produce a dictionary of its values.

        <br>

    2) ### ***path_getter.py:***
        There are many functions contained in this file. It's main purpose is to recursively get the file paths of all validation folds. However, it will also find the validation folds' history files and the indexes of the available configurations.
        
        <br>

    3) ### ***predicted_formatter.py:***
        This will convert prediction probabilities into indexed classes. This can be run manually, but the previous files will also run it if the indexed predictions do not exist.
        ***Example:*** 
        > python3 predicted_formatter.py -j my_config.json
        * ***Input:*** The configuration file. *(Optional)*
        * ***Output:*** An indexed alternative of every prediction file.
        
        <br>

    4) ### ***predicted_formatter.py:***
        This will convert prediction probabilities into indexed classes. This can be run manually, but the previous files will also run it if the indexed predictions do not exist.
        ***Example:*** 
        ```json
            {
                "data_path": "[path]/data/"
            }
        ```
        * ***data_path:*** The directory path of the data as a whole. This folder should contain the testing fold directories.
        
        <br>


    </details>

---
