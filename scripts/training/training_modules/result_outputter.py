from termcolor import colored
import pandas as pd
import csv
import os


def output_results(output_path, testing_subject, validation_subject, rotation, model_obj, history, time_elapsed, datasets, class_names):
    """ Output results from the trained model.
        
    Args:
        output_path (str): Where to output the results.
        testing_subject (str): The testing subject name.
        validation_subject (str): The validation subject name.
        
        rotation (int): The rotatiion index.
        model_obj (TrainingModel): The trained model.
        history (keras History): The history outputted by the fitting function.
        
        time_elapsed (double): The elapsed time from the fitting phase.
        datasets (dict): A dictionary of various values for the data-splits.
        class_names (list of str): The class names of the data.
    """
    # Check if the output paths exist
    file_prefix = f"{model_obj.model_type}_{rotation}_test_{testing_subject}_val_{validation_subject}"
    path_prefix = os.path.join(output_path, f'Test_subject_{testing_subject}', file_prefix)
    _create_folders(path_prefix, ['prediction', 'true_label', 'file_name', 'model'])
    
    # Save the model
    model_obj.model.save(f"{path_prefix}/model/{file_prefix}_{model_obj.model_type}.h5")
    
    # Save the history
    if history is not None:
        history = pd.DataFrame.from_dict(history.history)
        history.to_csv(f"{path_prefix}/{file_prefix}_history.csv")
    else:
        print(colored(f'Warning: The model history is empty for: {file_prefix}', 'yellow'))
    
    # Write the class names
    _metric_writer(
         f"{file_prefix}_class_names.csv", 
         [f"{class_names[idx]}, {idx}" for idx in range(len(class_names))], 
         path_prefix
    )
    
    # The values of the various model metrics and their paths to write to
    metrics = {
        # Training time
        f"{file_prefix}_time-total.csv":                        [time_elapsed],  
        
        # Predicted probability results
        f"prediction/{file_prefix}_val_predicted.csv":          model_obj.model.predict(datasets['validation']['ds']), 
        
        # True labels
        f"true_label/{file_prefix}_val_true_label.csv":         datasets['validation']['labels'], 
        f"true_label/{file_prefix}_test_true_label.csv":        datasets['testing']['labels'], 
        
        # True index 
        f'true_label/{file_prefix}_val_true_label_index.csv':   datasets['validation']['indexes'], 
        f'true_label/{file_prefix}_test_true_label_index.csv':  datasets['testing']['indexes'], 
        
        # Input file names (validation)
        f'file_name/{file_prefix}_val_file.csv':                datasets['validation']['files'], 
        f'file_name/{file_prefix}_test_file.csv':               datasets['testing']['files'] 
    }
    
    # If possible, write the metrics using the testing dataset
    if datasets['testing']['ds'] is None:
        print(colored(
            f"Non-fatal Error: evaluation was skipped for the test subject {testing_subject} and val subject {validation_subject}. " + 
            f"There were no files in the testing dataset.",
            'yellow'
        ))
    else:
        metrics[f"{file_prefix}_test_evaluation.csv"] =  model_obj.model.evaluate(datasets['testing']['ds'])          # The evaluation results
        metrics[f"prediction/{file_prefix}_test_predicted.csv"] =  model_obj.model.predict(datasets['testing']['ds']) # Predictions using the testing dataset
    
    # Write all metrics to file
    for metric in metrics:
        _metric_writer(metric, metrics[metric], path_prefix)
    print(colored(f"Finished writing results to file for {model_obj.model_type}'s testing subject {testing_subject} and validation subject {validation_subject}.\n", 'green'))


def _create_folders(path, names=None):
    """ Creates folder(s) if they do not exist.

    Args:
        path (str): Name of the path to the folder.
        names (list): The folder name(s). (Optional.)
    """
    if names is not None:
        for name in names:
            folder_path = os.path.join(path, name)
            if not os.path.exists(folder_path): os.makedirs(folder_path)
    else:
        if not os.path.exists(path): os.makedirs(path)

   
def _metric_writer(path, values, path_prefix):
    """ Writes some list to file.

    Args:
        path (str): A file path.
        values (list): A list of values.
        path_prefix (str): The prefix of the file name and directory.
    """
    with open(f"{path_prefix}/{path}", 'w', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        if path.endswith('predicted.csv'):
            for item in values:
                writer.writerow(item)
        else:
            for item in values:
                writer.writerow([item])