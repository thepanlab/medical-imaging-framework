from termcolor import colored
from tensorflow import keras
import pandas as pd
import image_parser
import numpy as np
import argparse
import json
import time
import os

# Hide tensorflow console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


def parse_json(default_config_file_name):
    """ 
        Reads in the configuration from a JSON file. 

        default_config_file_name: The default config input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-j', '--json', '--load_json',
        help='Load settings from a JSON file.',
        required=False,
        default=default_config_file_name
    )
    args = parser.parse_args()
    with open(args.json) as config_file:
        config = json.load(config_file)
    return config


def get_images(data_input, class_names):
    """
        Reads in a dictionary of arrays of data-paths.

        data_input: A dictionary of key-array form.
        return: A dictionary of key-array form, of all the images.
    """
    # To sort the input filenames by their index
    def sort_condition(filename):
        return int(filename.split('/')[-1].split('_')[0])
                
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
        image_paths[subject] = {}
        for dir in subdirs:
            dir = os.path.join(data_input[subject], dir)
            if not os.path.isdir(dir):
                raise Exception(colored(f"Error: Expected '{dir}' within '{data_input[subject]}' to be a directory path.", 'red'))
            subfiles = os.listdir(dir)
            if len(subfiles) == 0:
                print(colored(f"Warning: The data path '{dir}' is empty.", 'yellow'))
                continue
            
            # Iterate through every image path
            img_class = [c for c in class_names if c in dir.split('/')[-1]]
            if len(img_class) != 1:
                raise Exception(colored(f"Error: Expected '{dir}' to contain one class name. It had {len(img_class)}.", 'red'))
            image_paths[subject][img_class[0]] = []
            for file in subfiles:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    image_paths[subject][img_class[0]].append(os.path.abspath(os.path.join(dir, file)))
                else:
                    print(colored(f"Warning: Non-png or non-jpg file detected. '{file}'", 'yellow'))  
            image_paths[subject][img_class[0]].sort(key=sort_condition)  
    return image_paths
        

def read_models(model_input):
    """
        Reads in a dictionary of models.

        data_input: A dictionary of key-path form.
        return: A dictionary of key-model form.
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


def predict_images(image_paths, models, batch_size, class_names, image_config):
    """
        Predicts values of images for each given model.

        data: A dictionary of key-array form, containing images.
        models: A dictionary of key-model form.
        return: A key-model-array dictionary of prediction results.
    """   
    # Image args
    class_names = image_config['class_names'].split(',')
    target_height = int(image_config['target_height'])
    offset_height = int(image_config['offset_width'])
    offset_width = int(image_config['offset_width'])
    target_width = int(image_config['target_width'])
    do_cropping = image_config['do_cropping']
    channels = int(image_config['channels']) 
    use_mean = image_config['use_mean']
    mean = float(image_config['mean'])
    
    # Get label position
    for subject in image_paths:
        for label in image_paths[subject]:
            for file_name in image_paths[subject][label]:
                labels = [class_name for class_name in class_names if class_name in file_name]
                temp = os.path.abspath(file_name).split('.')
                label_position = temp[0].split('_').index(labels[0])
                break
            break
        break
    
    # Predict all items using every model
    prediction_results = {}
    timing_results = {}
    for model in models:
        prediction_results[model] = {}
        timing_results[model] = {}
        
        # Loop every subjects' label
        for subject in image_paths:
            prediction_results[model][subject] = {}
            timing_results[model][subject] = {}
            for label in image_paths[subject]:
                
                # Separate the images into batches
                img_slices = tf.data.Dataset.from_tensor_slices(image_paths[subject][label])
                img_map = img_slices.map(lambda x:image_parser.parse_image(
                    x,  mean, use_mean, class_names, label_position, channels, 
                    do_cropping,offset_height, offset_width, target_height, target_width
                ))
                img_batch = img_map.batch(batch_size, drop_remainder=False)

                # Add predictions to the array
                srt_cpu = time.process_time()
                srt_wal = time.time()
                pred = models[model].predict(img_batch)
                end_cpu = time.process_time()
                end_wal = time.time()
                timing_results[model][subject][label] = {'cpu': end_cpu-srt_cpu, 'wall': end_wal-srt_wal}
                prediction_results[model][subject][label] = pred
                print(colored(f"Successfully computed the predictions for: model '{model}' and input '{subject}' and label '{label}'", 'green'))
        print(' ')
    return prediction_results, timing_results


def output_results(prediction_results, timing_results, input_filepaths, class_names, out_vals, out_mets):
    """
        Output the prediction results to CSV file.

        prediction_results: A key-model-array dictionary of prediction results.
        output_directory: The directory to output the predictions to.
        return: None.
    """    
    # Output results to file
    columns_metrics = ['dirname', 'subject', 'true_label', 'true_label_index', 'wall_time', 'cpu_time']
    columns_predicted_labels = ['subject', 'filename', 'true_label', 'true_label_index', 'pred_label', 'pred_label_index', 'match']
    for model in prediction_results:
        metrics = pd.DataFrame(columns=columns_metrics)
        predicted_labels = pd.DataFrame(columns=columns_predicted_labels)
        for subject in prediction_results[model]:
            for label in prediction_results[model][subject]:
                
                # Print the predictions
                filename = os.path.join(out_vals, f"{subject}_{model}_{label}_predictions.csv")
                pd.DataFrame(prediction_results[model][subject][label]).to_csv(filename, index=False)
                print(colored(f"Successfully output the predictions for: model '{model}' and input '{subject}'", 'green'))
                
                # Get the metrics
                label_index = class_names.index(label)
                dirname = input_filepaths[subject][label][0].split('/')[-2]
                metrics = pd.concat([metrics, pd.DataFrame.from_dict({
                    'dirname': [dirname],
                    'subject': [subject],
                    'true_label': [label],
                    'true_label_index': [label_index],
                    'wall_time': [timing_results[model][subject][label]['wall']],
                    'cpu_time': [timing_results[model][subject][label]['cpu']]
                })], ignore_index=True)
                
                # Get the prediction results for every image
                for i in range(len(input_filepaths[subject][label])):
                    img_file = input_filepaths[subject][label][i].split('/')[-1]
                    predictions = list(prediction_results[model][subject][label][i])
                    max_index = predictions.index(max(predictions))
                    max_label = class_names[max_index]
                    predicted_labels = pd.concat([predicted_labels, pd.DataFrame.from_dict({
                        'subject': [subject],
                        'filename': [img_file],
                         'true_label': [label],
                         'true_label_index': [label_index],
                         'pred_label': [max_label],
                         'pred_label_index': [max_index],
                         'match': [label_index == max_index]                        
                    })], ignore_index=True)
        
        # Print the metrics
        filename = os.path.join(out_mets, f"{model}_metrics.csv")
        metrics.sort_values(by=['dirname']).to_csv(filename, index=False)
        print(colored(f"Successfully output the metrics for: model '{model}'", 'green'))
        
        # Print the predicted labels
        filename = os.path.join(out_mets, f"{model}_predicted_labels.csv")
        predicted_labels.sort_values(by=['subject']).to_csv(filename, index=False)
        print(colored(f"Successfully output the predicted labels for: model '{model}'\n", 'green'))
                 

def main(config=None):
    """ 
        The main body of the program.

        config: The input configuration, as a dictionary. (Optional)
    """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('prediction_config.json')

    # Check that output directories exist
    if not os.path.exists(config['prediction_output']): os.makedirs(config['prediction_output'])
    out_vals = os.path.join(config['prediction_output'], 'prediction_values')
    if not os.path.exists(out_vals): os.makedirs(out_vals)
    out_mets = os.path.join(config['prediction_output'], 'prediction_metrics')
    if not os.path.exists(out_mets): os.makedirs(out_mets)

    # Read in the input data
    class_names = config['image_settings']['class_names'].split(',')
    data = get_images(config["test_subject_data_input"], class_names)
    print(colored('Input images sucessfully read.', 'green'))
    
    # Read in the models
    models = read_models(config["model_input"])
    print(colored('Input models sucessfully read.\n', 'green'))

    # Predict the images
    prediction_results, timing_results = predict_images(data, models, config['batch_size'], class_names, config["image_settings"])

    # Output the results
    output_results(prediction_results, timing_results, data, class_names, out_vals, out_mets)
    

if __name__ == "__main__":
    main()
