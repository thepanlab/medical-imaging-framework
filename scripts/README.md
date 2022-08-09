# Scripts

* **confusion_matrix.py** - Creates confusion matrices for each outer loop iteration. Using the confusion matrices created an average and standard error confusion matrix is created as well. Each matrix is saved as a csv file in results.

    * The **conf_matrix_config.json** is a configuration file called in **confusion_matrix.py** that can be used to change the directories used as input/output and the name of the labels. 

* **learning_curve.py** - Using the history files produced, learning curved based on loss and accuracy are created. Each graph made is saved in results. 

    * The **learning_curve_config.json** is a configuration file called in **learning_curve.py** that can be used to change the directories used as input/output and several aspects of the plots produced.

        * Colors used for plotted lines
        * Line width
        * Resolution of saved plot image
        * Format to use to save the plot created
        * Font size of title and labels 
        * Font family used

* **roc_curve.py** -  Creates an ROC curve using the predicted values from the model and true lables for each inner loop iteration. Each graph made is saved in results.

    * The **roc_curve_config.json** is a configuration file called in **roc_curve.py** that can be used to change the directories used as input/output and several aspects of the plots produced. 
        
        * Colors used for plotted lines
        * Line width
        * Resolution of saved plot image
        * Format to use to save the plot created
        * Font size of title and labels 
        * Font family used

- To Do:
    - File managment
    - How to save, create new dirctories for each, named after each run?
    - Grabbing files 
    - More safe guards