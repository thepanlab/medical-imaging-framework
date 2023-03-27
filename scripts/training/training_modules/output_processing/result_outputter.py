from termcolor import colored
import fasteners
import pandas as pd
import json
import csv
import os


def output_results(output_path, testing_subject, rotation_subject, rotation, model_obj, history, time_elapsed, datasets, class_names, job_name, config_name, is_outer, rank):
    """ Output results from the trained model.
        
    Args:
        output_path (str): Where to output the results.
        testing_subject (str): The testing subject name.
        rotation_subject (str): The validation subject name.
        
        rotation (int): The rotation index.
        model_obj (TrainingModel): The trained model.
        history (keras History): The history outputted by the fitting function.
        
        time_elapsed (double): The elapsed time from the fitting phase.
        datasets (dict): A dictionary of various values for the data-splits.
        class_names (list of str): The class names of the data.
        
        job_name (str): The name of this config's job name.
        config_name (str): The name of this config's config (model) name.
        is_outer (bool): If this is of the outer loop.
    """
    # Check if the output paths exist
    if is_outer:
        file_prefix = f"{model_obj.model_type}_{rotation}_test_{testing_subject}"
    else:
        file_prefix = f"{model_obj.model_type}_{rotation}_test_{testing_subject}_val_{rotation_subject}"
    path_prefix = os.path.join(
        output_path, 
        f'Test_subject_{testing_subject}', 
        f'config_{job_name}_{config_name}',
        file_prefix
    )
    
    if rank is not None:
        with fasteners.InterProcessLock(os.path.join(output_path, 'output_lock.tmp')):
            _create_folders(path_prefix, ['prediction', 'true_label', 'file_name', 'model'])
    else:
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
    class_output = {class_names[idx]: idx for idx in range(len(class_names))}
    _metric_writer(
         f"{file_prefix}_class_names.json", 
         class_output, 
         path_prefix
    )
    
    # The values of the various model metrics and their paths to write to
    metrics = {
        # Training time
        f"{file_prefix}_time-total.csv": [time_elapsed],
    }
    
    # Inner loop file names
    if not is_outer:
        # Predicted probability results
        metrics[f"prediction/{file_prefix}_val_predicted.csv"] = model_obj.model.predict(datasets['validation']['ds'])
        
        # True labels
        metrics[f"true_label/{file_prefix}_test_true_label.csv"] = datasets['testing']['labels']
        metrics[f"true_label/{file_prefix}_val_true_label.csv"] = datasets['validation']['labels']
        
        # True index 
        metrics[f'true_label/{file_prefix}_true_label_index.csv'] = [class_output[l] for l in datasets['testing']['labels']]
        metrics[f'true_label/{file_prefix}_val_true_label_index.csv'] = [class_output[l] for l in datasets['validation']['labels']]
        
        # Input file name
        metrics[f'file_name/{file_prefix}_test_file.csv'] = datasets['testing']['files']
        metrics[f'file_name/{file_prefix}_val_file.csv'] =  datasets['validation']['files']  
    
    # Outer loop file names
    else:
        # Predicted probability results
        metrics[f"prediction/{file_prefix}_predicted.csv"] = model_obj.model.predict(datasets['testing']['ds'])
        
        # True labels
        metrics[f"true_label/{file_prefix}_true_label.csv"] = datasets['testing']['labels']
        
        # True index 
        metrics[f'true_label/{file_prefix}_label_index.csv'] = [class_output[l] for l in datasets['testing']['labels']]
        
        # Input file name
        metrics[f'file_name/{file_prefix}_file.csv'] = datasets['testing']['files']
    
    
    # If possible, write the metrics using the testing dataset
    if datasets['testing']['ds'] is None:
        print(colored(
            f"Non-fatal Error: evaluation was skipped for the test subject {testing_subject} and val subject {rotation_subject}. " + 
            f"There were no files in the testing dataset.",
            'yellow'
        ))
    else:
        metrics[f"{file_prefix}_test_evaluation.csv"] =  model_obj.model.evaluate(datasets['testing']['ds'])          # The evaluation results
        if not is_outer:
            metrics[f"prediction/{file_prefix}_test_predicted.csv"] =  model_obj.model.predict(datasets['testing']['ds']) # Predictions using the testing dataset
    
    # Write all metrics to file
    for metric in metrics:
        _metric_writer(metric, metrics[metric], path_prefix)
    print(colored(f"Finished writing results to file for {model_obj.model_type}'s testing subject {testing_subject} and validation subject {rotation_subject}.\n", 'green'))


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
        # Predicted values
        if path.endswith('predicted.csv'):
            writer = csv.writer(fp)
            for item in values:
                writer.writerow(item)
                
        # Class names
        elif path.endswith('_class_names.json'): 
            json.dump(values, fp)
            
            
        # Everything else
        else:
            writer = csv.writer(fp)
            for item in values:
                writer.writerow([item])