from termcolor import colored
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import image_parser
import argparse
import json
import os


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


def get_images(data_input):
    """
        Reads in a dictionary of arrays of data-paths.

        data_input: A dictionary of key-array form.
        return: A dictionary of key-array form, of all the images.
    """    
    # Check that the input paths exist
    for key in data_input:
        if not os.path.exists(data_input[key]):
            raise Exception(colored(
                f"The data input path '{data_input[key]}' does not exist", 
                'red'
            ))
        
    # Get all valid image paths
    image_paths = {}
    for key in data_input:
        if os.path.isdir(data_input[key]):
            files = os.listdir(data_input[key])
            if len(files) == 0:
                print(colored(f"Warning: The data path '{data_input[key]}' is empty.", 'yellow'))

        # If the given path is an image, add the file
        image_paths[key] = []
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                image_paths[key].append(os.path.abspath(os.path.join(data_input[key], file)))
            else:
                print(colored(f"Warning: Non-png or non-jpg file detected. '{file}'", 'yellow'))    
    return image_paths
    
    """
    # Finally, read in the images
    image_lists = {}
    image_config['class_names'] = image_config['class_names'].split(',')
    for key in image_paths:
        image_lists[key] = []
        for image_path in image_paths[key]: 
            labels = [class_name for class_name in image_config['class_names'] if class_name in image_path.lower().replace("%20", " ")]
            img, argmax = image_parser.parse_image(
                filename = image_path,
                mean = image_config['mean'],
                use_mean = image_config['use_mean'],
                class_names = image_config['class_names'],
                label_position = os.path.abspath(image_path).split('.')[0].split('_').index(labels[0]),
                channels = image_config['channels'],
                do_cropping = image_config['do_cropping'],
                offset_height = 241,
                offset_width = 181,
                target_height = 241,
                target_width = 181
            )
            image_lists[key].append(img)
    return image_lists """


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


def predict_images(image_paths, models, batch_size, image_config):
    """
        Predicts keyes of images for each given model.

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
    for key in image_paths:
        for file_name in image_paths[key]:
            labels = [class_name for class_name in class_names if class_name in file_name]
            temp = os.path.abspath(file_name).split('.')
            label_position = temp[0].split('_').index(labels[0])
            break
        break
    
    # Output a dictionary of key-model-array form. The arrays will contain the predicted labels in order.
    prediction_results = {}
    for model in models:
        prediction_results[model] = {}
        for key in image_paths:
            labels = [class_name for class_name in class_names if class_name in 
                      image_paths[key][0].lower().replace("%20", " ")]
            
            # Separate the images into batches
            img_slices = tf.data.Dataset.from_tensor_slices(image_paths[key])
            img_map = img_slices.map(lambda x:image_parser.parse_image(
                x,  mean, use_mean, class_names, label_position, channels, 
                do_cropping,offset_height, offset_width, target_height, target_width
            ))
            img_batch = img_map.batch(batch_size, drop_remainder=False)

            # Add predictions to the array
            pred = models[model].predict(img_batch)
            prediction_results[model][key] = pred
            print(colored(f"Successfully computed the predictions for: model '{model}' and input '{key}'", 'green'))
    return prediction_results


def output_predictions(prediction_results, output_directory):
    """
        Output the prediction results to CSV file.

        prediction_results: A key-model-array dictionary of prediction results.
        output_directory: The directory to output the predictions to.
        return: None.
    """
    # Output results to file
    output_directory = os.path.abspath(output_directory)
    for model in prediction_results:
        for key in prediction_results[model]:
            filename = os.path.join(output_directory, f"{key}_{model}_predictions.csv")
            pd.DataFrame(prediction_results[model][key]).to_csv(filename)
            print(colored(f"Successfully output the predictions for: model '{model}' and input '{key}'", 'green'))
                 

def main(config=None):
    """ 
        The main body of the program.

        config: The input configuration, as a dictionary. (Optional)
    """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('prediction_config.json')

    # Check that output directory exists
    if not os.path.exists(config['prediction_output']):
        os.makedirs(config['prediction_output'])

    # Read in the input data
    data = get_images(config["data_input"])
    print(colored('Input images sucessfully read.', 'green'))
    
    # Read in the models
    models = read_models(config["model_input"])
    print(colored('Input models sucessfully read.', 'green'))

    # Predict the images
    prediction_results = predict_images(data, models, config['batch_size'], config["image_settings"])

    # Output the predictions
    output_predictions(prediction_results, config["prediction_output"])
            

if __name__ == "__main__":
    main()
