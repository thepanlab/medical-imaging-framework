from util.get_config import parse_json
from training import image_parser
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

        data_input: A dictionary of key-array form.
        return: A dictionary of key-array form, of all the images.
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
        for dir in subdirs:
            dir = os.path.join(data_input[subject], dir)
            if not os.path.isdir(dir):
                raise Exception(colored(f"Error: Expected '{dir}' within '{data_input[subject]}' to be a directory path.", 'red'))
            subfiles = os.listdir(dir)
            if len(subfiles) == 0:
                print(colored(f"Warning: The data path '{dir}' is empty.", 'yellow'))
                continue
            for file in subfiles:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    image_paths[subject].append(os.path.abspath(os.path.join(dir, file)))
                else:
                    print(colored(f"Warning: Non-png or non-jpg file detected. '{file}'", 'yellow'))
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
        for file_name in image_paths[subject]:
            labels = [class_name for class_name in class_names if class_name in file_name]
            temp = os.path.abspath(file_name).split('.')
            label_position = temp[0].split('_').index(labels[0])
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
            img_map = img_slices.map(lambda x:image_parser.parse_image(
                x,  mean, use_mean, class_names, label_position, channels, 
                do_cropping,offset_height, offset_width, target_height, target_width
            ))
            img_batch = img_map.batch(batch_size, drop_remainder=False)

            # Get the computation time
            srt = time.time()
            pred = models[model].predict(img_batch)
            timing_results[model][subject] = time.time() - srt
            
            # Add predictions to the array
            prediction_results[model][subject] = pred
            print(colored(f"Successfully computed the predictions for: model '{model}' and input '{subject}'", 'green'))
        print(' ')
    return prediction_results, timing_results, label_position


def output_results(prediction_results, timing_results, label_pos, input_filepaths, class_names, out_vals, out_mets):
    """
        Output the prediction results to CSV file.

        prediction_results: A key-model-array dictionary of prediction results.
        output_directory: The directory to output the predictions to.
        return: None.
    """    
    # Output results to file
    columns_predicted_labels = ['subject', 'filename', 'true_label', 'true_label_index', 'pred_label', 'pred_label_index', 'match']
    for model in prediction_results:
        predicted_labels = pd.DataFrame(columns=columns_predicted_labels)
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
            if not os.path.exists(os.path.join(dirpath, 'prediction')): os.makedirs(os.path.join(dirpath, 'prediction'))
            if not os.path.exists(os.path.join(dirpath, 'true_label')): os.makedirs(os.path.join(dirpath, 'true_label'))
            if not os.path.exists(os.path.join(dirpath, 'file_name')): os.makedirs(os.path.join(dirpath, 'file_name'))
            prefix = f"{model}_{model_i}_test_{subject}_test"
                
            # Print the prediction probability values
            preds = pd.DataFrame(prediction_results[model][subject])
            filename = os.path.join(dirpath, f"prediction/{prefix}_predicted.csv")
            preds.to_csv(filename, index=False, header=False)
            print(colored(f"\t Wrote the predictions.", 'cyan'))
            
            # Print the prediction index values
            preds = preds.idxmax(axis=1)
            filename = os.path.join(dirpath, f"prediction/{prefix}_predicted_index.csv")
            preds.to_csv(filename, index=False, header=False)
            print(colored(f"\t Wrote the indexed predictions.", 'cyan'))
            
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
            
            # Print the true label index values
            files = pd.DataFrame(input_filepaths[subject])
            filename = os.path.join(dirpath, f"file_name/{prefix}_file.csv")
            files.to_csv(filename, index=False, header=False)
            print(colored(f"\t Wrote the image file paths.", 'cyan'))
            
            # Print the true label index values
            time = pd.DataFrame([timing_results[model][subject]])
            filename = os.path.join(dirpath, f"{prefix}_time_total.csv")
            time.to_csv(filename, index=False, header=False)
            print(colored(f"\t Wrote the timing.", 'cyan'))
            
            # Get the prediction info for every image
            for i in range(len(input_filepaths[subject])):
                img_file = input_filepaths[subject][i].split('/')[-1]
                predictions = list(prediction_results[model][subject][i])
                max_index = predictions.index(max(predictions))
                max_label = class_names[max_index]
                label = trues.values[i][0]
                predicted_labels = pd.concat([predicted_labels, pd.DataFrame.from_dict({
                    'subject': [subject],
                    'filename': [img_file],
                    'true_label': [label],
                    'true_label_index': [class_names[label]],
                    'pred_label': [max_label],
                    'pred_label_index': [max_index],
                    'match': [class_names[label] == max_index]                        
                })], ignore_index=True)
            print(colored(f"\t Got the prediction info.", 'cyan'))
            print(colored(f"\t Finished writing the predictions.", 'green'))
        
        # Print the predicted labels
        filename = os.path.join(out_mets, f"{model}_prediction_info.csv")
        predicted_labels.sort_values(by=['subject']).to_csv(filename, index=False)
        print(colored(f"Successfully output the predicted labels for: model '{model}'\n", 'green'))
                 

def main(config=None):
    """ 
        The main body of the program.

        config: The input configuration, as a dictionary. (Optional)
    """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('./results_processing/prediction/prediction_config.json')

    # Check that output directories exist
    if not os.path.exists(config['prediction_output']): os.makedirs(config['prediction_output'])
    out_vals = os.path.join(config['prediction_output'], 'prediction_values')
    if not os.path.exists(out_vals): os.makedirs(out_vals)
    out_mets = os.path.join(config['prediction_output'], 'tabled_prediction_info')
    if not os.path.exists(out_mets): os.makedirs(out_mets)

    # Read in the input data
    class_names = config['image_settings']['class_names'].split(',')
    data = get_images(config["test_subject_data_input"], class_names)
    print(colored('Input images sucessfully read.', 'green'))
    
    # Read in the models
    models = read_models(config["model_input"])
    print(colored('Input models sucessfully read.\n', 'green'))

    # Predict the images
    prediction_results, timing_results, label_pos = predict_images(data, models, config['batch_size'], class_names, config["image_settings"])

    # Output the results
    output_results(prediction_results, timing_results, label_pos, data, class_names, out_vals, out_mets)
    

if __name__ == "__main__":
    main()
