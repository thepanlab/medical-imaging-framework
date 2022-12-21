# **Util**

- - -
+ ##  ***get_config.py:***
    This will take in a configuration file path and produce a dictionary of its values.

    <br>

+ ##  ***path_getter.py:***
    There are many functions contained in this file. It's main purpose is to recursively get the file paths of all validation folds. However, it will also find the validation folds' history files and the indexes of the available configurations.
    
    <br>

+ ##  ***predicted_formatter.py:***
    This will convert prediction probabilities into indexed classes. This can be run manually, but the previous files will also run it if the indexed predictions do not exist.
    ***Example:*** 
    > python3 predicted_formatter.py -j my_config.json
    * ***Input:*** The configuration file. *(Optional)*
    * ***Output:*** An indexed alternative of every prediction file.
    
    <br>

+ ##  ***predicted_formatter.py:***
    This will convert prediction probabilities into indexed classes. This can be run manually, but the previous files will also run it if the indexed predictions do not exist.
    ***Example:*** 
    ```json
        {
            "data_path": "[path]/data/"
        }
    ```
    * ***data_path:*** The directory path of the data as a whole. This folder should contain the testing fold directories.
    
    <br>

---
