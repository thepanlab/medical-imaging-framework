from results_processing.tabled_prediction_info import tabled_prediction_info
from training.training_modules.image_processing.image_parser import *
from util.get_config import parse_json
from termcolor import colored
from tensorflow import keras
import pandas as pd
import time
import os

# Hide tensorflow console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


def get_images(data_input, class_names):
    """
        Reads in a dictionary of arrays of data-paths.

    Args:
        data_input (dict): A dictionary of key-array form.
        class_names (list of str): A list of classes/labels.
        
    Reeturns:
        (dict) A dictionary of key-array form, of all the images.
    """                
    # Get all valid image paths
    image_paths = {}
    for subject in data_input:     
        # Check if valid location
        if not os.path.isdir(data_input[subject]):
            raise Exception(colored(f"Error: Expected '{data_input[subject]}' to be a directory path.", 'red'))
        subdirs = os.listdir(data_input[subject])
        if len(subdirs) == 0:
            print(colored(f"Warning: The data path '{data_input[subject]}' is empty.", 'yellow'))
                
        # Get all of the subject's image subdirectories
        image_paths[subject] = []
        for subsubdir in subdirs:
            subsubdir = os.path.join(data_input[subject], subsubdir)
            
            # If this level contains images
            if subsubdir.endswith((".png", ".jpg", ".jpeg")):
                image_paths[subject].append(subsubdir)
            else:
                subfiles = os.listdir(subsubdir)
                if len(subfiles) == 0:
                    print(colored(f"Warning: The data path '{subsubdir}' is empty.", 'yellow'))
                    continue
                
                # If outer loop, get images on this level
                for subfile in subfiles:
                    subfile = os.path.abspath(os.path.join(subsubdir, subfile))
                    if subfile.endswith((".png", ".jpg", ".jpeg")):
                        image_paths[subject].append(subfile)
                    else:
                        print(colored(f"Warning: Non-png or non-jpg file detected. '{subfile}'", 'yellow'))
    return image_paths
        

def read_models(model_input):
    """
        Reads in a dictionary of models.

    Args:
        model_input: An array of model paths.
        
    Return:
        A dictionary of key-model form.
    """
    # Check that the input models exist
    for model in model_input:
        if not os.path.exists(model_input[model]):
            raise Exception(colored(f"The model input path '{model}' does not exist", 'red'))

    # Output a dictionary of models
    model_output = {}
    for model in model_input:
        model_output[model] = keras.models.load_model(os.path.abspath(model_input[model]))
    return model_output


def predict_images(image_paths, models, config, class_names):
    """ Predicts values of images for each given model.

    Args:
        image_paths (dict): A dictionary of key-array form, containing images.
        models (dict): A dictionary of key-model form.
        batch_size (int): The size of the prediction subsets.
        config (dict): The program configuration.
        class_names (str): Names of the prediction classes.

    Returns:
        tuple: The prediction results and timing results
    """
    tf.config.run_functions_eagerly(True)
    
    # Image args
    class_names = config['image_settings']['class_names']
    offset_height = config['image_settings']['offset_width']
    offset_width = config['image_settings']['offset_width']
    target_height = config['image_settings']['target_height']
    target_width = config['image_settings']['target_width']
    do_cropping = config['image_settings']['do_cropping']
    channels = config['image_settings']['channels']
    
    # Get label position
    label_position = -1
    if config['use_true_labels']:
        for subject in image_paths:
            for file_name in image_paths[subject]:
                labels = [class_name for class_name in class_names if class_name in file_name]
                temp = os.path.abspath(file_name).split('/')[-1].split('.')[0]
                try:
                    label_position = temp.split('_').index(labels[0])
                except:
                    raise ValueError(colored("Error: Class not found in image name. Are the classes correctly spelled or captialized?", 'red'))
                break
            break
    
    # Predict all items using every model
    prediction_results = {}
    timing_results = {}
    for model in models:
        prediction_results[model] = {}
        timing_results[model] = {}
        
        # Loop through every subject
        for subject in image_paths:
            prediction_results[model][subject] = {}
            timing_results[model][subject] = {}
                
            # Separate the images into batches
            img_slices = tf.data.Dataset.from_tensor_slices(image_paths[subject]) 
            if config['use_true_labels']:
                tout = [tf.float32, tf.int64]
            else:
                tout = [tf.float32]
            img_map = img_slices.map(lambda x: tf.py_function(
                func=parse_image,
                inp=[
                    x,  
                    class_names, 
                    channels, 
                    do_cropping,  
                    offset_height, 
                    offset_width, 
                    target_height,
                    target_width, 
                    label_position,
                    config['use_true_labels'],
                ],
                Tout=tout
            ))
            """ Non-eager executing method
            img_map = img_slices.map(lambda x: parse_image(
                x,                  
                class_names, 
                channels, 
                do_cropping,  
                offset_height, 
                offset_width, 
                target_height,
                target_width, 
                label_position,
                config['use_true_labels']
            ))
            """
            img_batch = img_map.batch(config['batch_size'], drop_remainder=False)
            
            # Get the computation time
            print(colored(f"Beginning prediction for {len(image_paths[subject])} images.", 'yellow'))
            srt = time.time()
            pred = models[model].predict(img_batch)   #TODO         
            
            timing_results[model][subject] = time.time() - srt
            print(colored("Finished prediction.", 'cyan'))
            
            # Add predictions to the array
            prediction_results[model][subject] = pred
            print(colored(f"Successfully computed the predictions for: model '{model}' and input '{subject}'", 'green'))
        print(' ')
    return prediction_results, timing_results


def output_results(config, prediction_results, timing_results, input_filepaths, class_names, out_vals):
    """ Output the prediction results to CSV file.

    Args:
        config (dict): The program configuration.
        prediction_results (dict): A key-model-array dictionary of prediction results.
        timing_results (dict): The times of predictions.
        input_filepaths (dict): The true values, (images paths.)
        class_names (str): Names of the prediction classes.
        out_vals (str): Directory output path of the values.
    """
    
    # Output results to file
    for model in prediction_results:
        for subject in prediction_results[model]:
                
            # The files will stored in a particular context
            model_i = list(prediction_results.keys()).index(model)+1
            subject_i = list(prediction_results[model].keys()).index(subject)
            dirpath = os.path.join(
                out_vals,
                f"Test_subject_{subject}/config_{model_i}_{model}/{model}_{subject_i}_test_{subject}/"
            )
            print(colored(f"For model '{model}' and input '{subject}':", 'magenta'))
            
            # Make output directories
            if config['use_true_labels']:
                if not os.path.exists(os.path.join(dirpath, 'true_label')): 
                    os.makedirs(os.path.join(dirpath, 'true_label'))
            if not os.path.exists(os.path.join(dirpath, 'prediction')): 
                os.makedirs(os.path.join(dirpath, 'prediction'))
            if not os.path.exists(os.path.join(dirpath, 'file_name')): 
                os.makedirs(os.path.join(dirpath, 'file_name'))
            prefix = f"{model}_{model_i}_test_{subject}_test"
                
            # Print the prediction probability values
            preds = pd.DataFrame(prediction_results[model][subject])
            filename = os.path.join(dirpath, f"prediction/{prefix}_predicted.csv")
            preds.to_csv(filename, index=False, header=True)
            print(colored(f"\t Wrote the predictions.", 'cyan'))
            
            # Print the prediction index values
            preds = preds.idxmax(axis=1)
            filename = os.path.join(dirpath, f"prediction/{prefix}_predicted_index.csv")
            preds.to_csv(filename, index=False, header=False)
            print(colored(f"\t Wrote the indexed predictions.", 'cyan'))
            
            if config['use_true_labels']:
                # Print the true labels
                trues = pd.DataFrame(i for file in input_filepaths[subject] for i in file.split('/')[-1].split('.')[0].split('_') if i in class_names)
                filename = os.path.join(dirpath, f"true_label/{prefix}_true_label.csv")
                trues.to_csv(filename, index=False, header=False)
                print(colored(f"\t Wrote the true values.", 'cyan'))
            
                # Print the true label index values
                trues = pd.DataFrame(class_names.index(label[0]) for label in trues.values)
                filename = os.path.join(dirpath, f"true_label/{prefix}_true_label_index.csv")
                trues.to_csv(filename, index=False, header=False)
                print(colored(f"\t Wrote the indexed true values.", 'cyan'))
            
            # Print the file paths
            files = pd.DataFrame(input_filepaths[subject])
            filename = os.path.join(dirpath, f"file_name/{prefix}_file.csv")
            files.to_csv(filename, index=False, header=False)
            print(colored(f"\t Wrote the image file paths.", 'cyan'))
            
            # Print the timing
            time = pd.DataFrame([timing_results[model][subject]])
            filename = os.path.join(dirpath, f"{prefix}_time_total.csv")
            time.to_csv(filename, index=False, header=False)
            print(colored(f"\t Wrote the timing.", 'cyan'))
                 

def main(config=None):
    """ The main body of the program.

    Args:
        config (dict, optional): The input configuration, as a dictionary. Defaults to None.
    """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('./results_processing/prediction/prediction_config.json')
        
    # Set eager execution
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.config.run_functions_eagerly(True)

    # Check that output directories exist
    if not os.path.exists(config['prediction_output']): os.makedirs(config['prediction_output'])
    out_vals = os.path.join(config['prediction_output'], 'prediction_values')
    if not os.path.exists(out_vals): os.makedirs(out_vals)
    out_mets = os.path.join(config['prediction_output'], 'tabled_prediction_info')
    if not os.path.exists(out_mets): os.makedirs(out_mets)

    # Read in the input data
    class_names = config['image_settings']['class_names']
    data = get_images(config["test_subject_data_input"], class_names)
    print(colored('Input images sucessfully read.', 'green'))
    
    # Read in the models
    models = read_models(config["model_input"])
    print(colored('Input models sucessfully read.\n', 'green'))

    # Predict the images
    prediction_results, timing_results = predict_images(data, models, config, class_names)

    # Output the results
    output_results(config, prediction_results, timing_results, data, class_names, out_vals)
    
    # Output tabled info
    if config['output_tabled_info']:
        table_config = {
            "data_path": out_vals,
            "output_path": out_mets,
            "use_true_labels": config['use_true_labels'],
            "label_types": {str(class_names.index(label)): label for label in class_names},
            "is_outer": config['is_outer']
        }
        tabled_prediction_info.main(table_config)
        print(colored('Successfully printed the tabeled info.', 'green'))
    

if __name__ == "__main__":
    main()
