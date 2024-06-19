# **Util**
<ul> 
    This module contains various scripts that are used by the "results_processing" and "training" modules. The submodules within help to find files, process file metadata, and format input data.
</ul> <br>

- - -
+ ##  ***get_config.py:***
    This will take in a configuration file path and produce a dictionary of its values. The path must be given to in the prompt, if not already specified. Empty responses will use the default configuration file location.

    <br>

+ ##  ***path_getter.py:***
    There are many functions contained in this file. It's main purpose is to recursively get the file paths of all validation folds. However, it will also find the validation folds' history files, the indexes of the available configurations, and determine whether a given path belongs to the outer loop.
    
    <br>

+ ## ***Predicted Formatter***
    <ul> 
        Converts probabilities into indexes by finding the maximum liklihood of each row in a given csv file.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    

  + ###  ***predicted_formatter.py:***
      This will convert prediction probabilities into indexed classes. This can be run manually, but the previous files will also run it if the indexed predictions do not exist.
      ***Example:*** 
      > python3 predicted_formatter.py -j my_config.json

      * ***Input:*** The configuration file. *(Optional)*
      * ***Output:*** An indexed alternative of every prediction file.
      * ***predicted_formatter_config.json:***
        ```json
            {
                "data_path": "[path]/data/"
            }
        ```
        * ***data_path:*** The directory path of the data as a whole. This folder should contain the testing fold directories.
      
    </details>
    
<br>

+ ## ***Truth Formatter***
    <ul> 
        Converts indexes into a more readable format from the labels, if there is a mismatch between the values.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

  + ###  ***truth_formatter.py:***
      This will convert true labels into indexed classes. This should be run manually.
      ***Example:*** 
      > python3 truth_formatter.py -j my_config.json

      * ***Input:*** The configuration file. *(Optional)*
      * ***Output:*** An indexed alternative of every prediction file.
      * ***truth_formatter_config.json:***
        ```json
            {
                "data_path": "[path]/data/",
                "label_types": {"A": 0, "B":1},
                "is_outer": true
            }
        ```
        * ***data_path:*** The directory path of the data as a whole. This folder should contain the testing fold directories.
        * ***label_types:*** A dictionary of labels and their indexes.
        * ***is_outer:*** Whether the data is of the outer loop or not.
      
      <br>
    
    </details>

---
