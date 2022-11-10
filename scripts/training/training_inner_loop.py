from random import seed, shuffle
from termcolor import colored
from time import perf_counter
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import csv

from util.get_config import parse_training_configs
from training.inner_modules import *
from training.image_parser import *


""" Printing functions ---------------------------------------- """
def show_gpu_list():
    """ This will print a list of the available GPUs on the current system. """
    gpus = tf.config.list_physical_devices('GPU')
    print(colored(f"\n-----------------------------------------", 'magenta'))
    print(colored("GPU list:", 'magenta'))
    for gpu in gpus:
        print(colored(f"Name: {gpu.name}  | Type: {gpu.device_type}", 'cyan'))


""" Training functions ---------------------------------------- """
def generate_folds(subject_list, test_subject, in_rotations, do_shuffle):
    """ Generates folds for the subject.
        
        -- Input Parameters ------------------------
        subject_list (list of str): A list of subject names.
        test_subject (str): The current test subject name.
        in_rotations (int or 'all'): How many rotations were specified in the configuration.
        do_shuffle (bool): If the fold list should be shuffled or not.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of dict): A list of folds, containing the subjects for testing, validation, and training.
        (int): The number of rotations for the training loop.
        --------------------------------------------
    """
    folds = []
    
    # For this test subject, find all combinations for the testing data
    i = subject_list.index(test_subject)
    for j, validation_subject in enumerate(subject_list):
        if i != j:
            subject_fold = {'training': [], 'validation': [validation_subject], 'testing': [test_subject]}
            for k, training_subject in enumerate(subject_list):
                if (i != k) and (j != k):
                    subject_fold['training'].append(training_subject)
            folds.append(subject_fold)
    
    # Shuffle the data and get the number of training rotations
    if do_shuffle:
        shuffle(folds)
    if (in_rotations == 'all') or (in_rotations > len(folds)):
        return folds, len(folds)
    return folds, in_rotations
    

def get_files(input_path):
    """ Gets all of the input paths of the image data.
        
        -- Input Parameters ------------------------
        input_path (str): A path to some directory.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of str): A list of image paths.
        --------------------------------------------
    """
    # See if it exists
    if not os.path.isdir(input_path):
        raise Exception(colored(f"Error: '{input_path}' is not a valid input path.", 'red'))
    
    # TODO: move this
    def flatten_dir(path, files):
        """ Gets the paths of ALL images within a directory and its subdirectories.

        Args:
            path (str): A path to some directory.
            files (list of str): A list of paths to images.
        """
        for item in os.listdir(os.path.abspath(path)):
            full_path = os.path.join(path, item)
            if os.path.isfile(full_path):
                if full_path.endswith((".png", ".jpg", ".jpeg")):
                    files.append(full_path)
                else:
                    print(colored(f"Warning: Non-image file detected '{full_path}'"))
            else:
                flatten_dir(full_path, files)
    
    # Search through the first level, of subdirectories (1)
    files = []
    flatten_dir(input_path, files)
                    
    # Shuffle the list
    files.sort()
    shuffle(files)
    print(colored('Finished getting the image paths.', 'green'))
    return files
 

def get_indexes(files, class_names, subject_list):
    """ Gets the indexes of classes, and the subjects within the image file names.
        
        -- Input Parameters ------------------------
        files (list of str): List of input image paths.
        class_names (list of str): List of class names.
        subject_list (list of str): List of subject names
        --------------------------------------------
        
        -- Returns ---------------------------------
        (dict of lists): A dictionary containing all labels, indexes, and subjects.
        (int): The label position.
        --------------------------------------------
    """
    indexes = {'labels': [], 'idx': [], 'subjects': []}
    label_position = None
    
    for file in files:
        
        # Get the proper filename to parse
        formatted_name = file.lower().replace("%20", " ").split('/')[-1].split('.')[:-1]
        formatted_name = ''.join(formatted_name)
        formatted_name = formatted_name.split('_')
        
        # Get the image label
        labels = [c for c in class_names if c in formatted_name]
        if len(labels) != 1:
            raise Exception(colored(f"Error: {len(labels)} labels found for '{file}'. There should be only one.", 'red'))
        
        # Get index of the label in the class name list
        idx = class_names.index(labels[0])
            
        # Get the image subject
        subjects = [s for s in subject_list if s in formatted_name]
        if len(subjects) != 1:
            raise Exception(colored(f"Error: {len(subjects)} subjects found for '{file}'. There should be only one.", 'red'))
        
        # Get the position of the label in the string
        if not label_position:
            label_position = formatted_name.index(labels[0])
            
        # Add the values
        indexes['labels'].append(labels[0])
        indexes['idx'].append(idx)
        indexes['subjects'].append(subjects[0])
        
    print(colored('Finished finding the label indexes.', 'green'))
    return indexes, label_position
        

def training_prep(config, test_subject):
    """ Gets some parameters needing for training
        
        -- Input Parameters ------------------------
        config (dict): List of input image paths.
        test_subject (str): List of class names.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of str): A list of file paths.
        (list of dict): A list of fold partitions.
        (int): The number of rotations to train with.
        (dict of lists): A dictionary of lists containing various indexes.
        (int): Position in the file name where the label exists.
        --------------------------------------------
    """
    # Set the seed
    seed(config['seed'])
    
    # Let the subjects be of the same case
    config['subject_list'] = [s.lower() for s in config['subject_list']]
    
    # The files to train with and info about its contents
    files = get_files(config['data_input_directory'])
    indexes, label_position = get_indexes(files, config['class_names'], config['subject_list'])
    
    # Generate training folds
    folds, rotations = generate_folds(config['subject_list'], test_subject, config['rotations'], config['shuffle_the_folds'])
    return files, folds, rotations, indexes, label_position


def create_model(hyperparameters, model_type, target_height, target_width, class_names):
    """ Creates and prepares a model for training.
        
        -- Input Parameters ------------------------
        hyperparameters (dict): The configuration's hyperparameters.
        model_type (str): Type of model to create.
        target_height (int): Height of input.
        target_width (int): Width of input.
        class_names (list of str): A list of classes.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (keras Model): The prepared keras model.
        (str): The name of the model type.
        --------------------------------------------
    """
    model_list = {
        "resnet_50": keras.applications.resnet50.ResNet50, 
        "resnet_VGG16": keras.applications.VGG16, 
        "InceptionV3": keras.applications.InceptionV3,
        "ResNet50V2": keras.applications.ResNet50V2, 
        "Xception": keras.applications.Xception
    }

    # Catch if the model is not in the model list
    if model_type not in model_list:
        print(colored(f"Warning: Model '{model_type}' not found in the list of possible models: {list(model_list.keys())}"))
        model_type = 'resnet_50'
        
    # Get the model base
    base_model = model_list[model_type](
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(target_height, target_width, hyperparameters['channels']),
        pooling=None
    )

    # Return the prepared model
    avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
    out = keras.layers.Dense(len(class_names), activation="softmax")(avg)
    model = keras.models.Model(inputs=base_model.input, outputs=out)
    
    # Create optimizer and add to model
    optimizer = keras.optimizers.SGD(
        lr=hyperparameters['learning_rate'], 
        momentum=hyperparameters['momentum'], 
        nesterov=True, 
        decay=hyperparameters['decay']
    )
    model.compile(
        loss="sparse_categorical_crossentropy", 
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model, model_type


def output_results(output_path, testing_subject, validation_subject, rotation, model, model_type, history, time_elapsed, datasets, class_names):
    """ Output results from the trained model.
        
        -- Input Parameters ------------------------
        output_path (str): Where to output the results.
        testing_subject (str): The testing subject name.
        validation_subject (str): The validation subject name.
        rotation (int): The rotatiion index.
        model (keras Model): The trained model.
        model_type (str): The model name.
        history (keras History): The history outputted by the fitting function.
        time_elapsed (double): The elapsed time from the fitting phase.
        datasets (dict): A dictionary of various values for the data-splits.
        class_names (list of str): The class names of the data.
        --------------------------------------------
    """
    # Check if the output paths exist
    file_prefix = f"{model_type}_{rotation}_test_{testing_subject}_val_{validation_subject}"
    path_prefix = os.path.join(output_path, f'Test_subject_{testing_subject}', file_prefix)
    create_folders(path_prefix, ['prediction', 'true_label', 'file_name', 'model'])
    
    # Save the model
    model.save(f"{path_prefix}/model/{file_prefix}_{model_type}.h5")
    
    # Save the history
    history = pd.DataFrame.from_dict(history.history)
    history.to_csv(f"{path_prefix}/{file_prefix}_history.csv")
    
    # Write the class names
    with open(f"{path_prefix}/{file_prefix}_class_names.csv", 'w', encoding='utf-8') as fp:
        for idx in range(len(class_names)):
            row = f"{class_names[idx]}, {idx}\n"
            fp.write(row)
    
    # Save the various model metrics
    metrics = {
        f"{file_prefix}_time-total.csv": [time_elapsed],  # Training time
        
        f"prediction/{file_prefix}_val_predicted.csv":  model.predict(datasets['validation']['ds']), # Predicted probability results
        
        f"true_label/{file_prefix}_val_true_label.csv":  datasets['validation']['labels'], # True labels (validation)
        f"true_label/{file_prefix}_test_true_label.csv": datasets['testing']['labels'],    # True labels (testing)
        
        f'true_label/{file_prefix}_val_true_label_index.csv':  datasets['validation']['indexes'], # True index (validation)
        f'true_label/{file_prefix}_test_true_label_index.csv': datasets['testing']['indexes'],    # True index (testing)
        
        f'file_name/{file_prefix}_val_file.csv':  datasets['validation']['files'], # Input file names (validation)
        f'file_name/{file_prefix}_test_file.csv': datasets['testing']['files']     # Input file names (testing)
    }
    
    # If possible, write the metrics using the testing dataset
    if datasets['testing']['ds'] is None:
        print(colored(
            f"Non-fatal Error: evaluation was skipped for the test subject {testing_subject} and val subject {validation_subject}. " + 
            f"There were no files in the testing dataset.",
            'yellow'
        ))
    else:
        metrics[f"{file_prefix}_test_evaluation.csv"] =  model.evaluate(datasets['testing']['ds'])          # The evaluation results
        metrics[f"prediction/{file_prefix}_test_predicted.csv"] =  model.predict(datasets['testing']['ds']) # Predictions using the testing dataset
    
    # Write metrics iterively
    for metric in metrics:
        with open(f"{path_prefix}/{metric}", 'w', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            if metric.endswith('predicted.csv'):
                for item in metrics[metric]:
                    writer.writerow(item)
            else:
                for item in metrics[metric]:
                    writer.writerow([item])
    print(colored(f"Finished writing results to file for {model_type}'s testing subject {testing_subject} and validation subject {validation_subject}.\n", 'green'))
        

def training_loop(config, test_subject, files, folds, rotations, indexes, label_position):
    """ Creates a model, trains it, and writes its outputs.
        
        -- Input Parameters ------------------------
        config (dict): The input configuration.
        test_subject (str): The name of the testing subject.
        files (list of str): A list of filepaths to images.
        folds (list of dict): A list of fold partitions.
        rotations (int): the number of rotations to perform.
        indexes (dict of lists): A list of true indexes.
        label_position (int): Location in the filename of the label.
        --------------------------------------------
    """
    print(colored(f'Beginning the training loop for {test_subject}.', 'green'))
    
    # Train for every rotation specified
    for rotation in range(rotations):
        print(colored(f'-- Rotation {rotation}/{rotations} for {test_subject} ---------------------------------------', 'magenta'))
        datasets = {
            'testing':    {'files': [], 'indexes': [], 'labels': [], 'ds': None},
            'training':   {'files': [], 'indexes': [], 'labels': [], 'ds': None},
            'validation': {'files': [], 'indexes': [], 'labels': [], 'ds': None}
        }
        
        # The target testing and validation subjects
        validation_subject = folds[rotation]['validation'][0]
        testing_subject = folds[rotation]['testing'][0]

        # Fill out the datasets' information
        for index, file in enumerate(files):
            dataset = ''
            if indexes['subjects'][index] == validation_subject:
                dataset = 'validation'
            elif indexes['subjects'][index] == testing_subject:
                dataset = 'testing'
            else:
                dataset = 'training'
            datasets[dataset]['files'].append(file)
            datasets[dataset]['indexes'].append(index)
            datasets[dataset]['labels'].append(indexes['labels'][index])
            
        # Get the datasets for each phase
        for dataset in datasets:
            if not datasets[dataset]['files']:
                continue
            ds = tf.data.Dataset.from_tensor_slices(datasets[dataset]['files'])
            ds_map = ds.map(lambda x:parse_image(
                x,                                                  # filename
                config['hyperparameters']['mean'],                  # mean
                config['hyperparameters']['use_mean'],              # use_mean
                config['class_names'],                              # class_names
                label_position,                                     # label_position
                config['hyperparameters']['channels'],              # channels
                config['hyperparameters']['do_cropping'],           # do_cropping
                config['hyperparameters']['cropping_position'][0],  # offset_height
                config['hyperparameters']['cropping_position'][1],  # offset_width
                config['target_height'],                            # target_height
                config['target_width']                              # target_width
            ))
            datasets[dataset]['ds'] = ds_map.batch(config['hyperparameters']['batch_size'], drop_remainder=False)
            
        # Create the model to train
        model, model_type = create_model(
            config['hyperparameters'],
            config['selected_model_name'], 
            config['target_height'], 
            config['target_width'], 
            config['class_names']
        )
        
        # Initalize early stopping
        early_stopping_callbacks = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=config['hyperparameters']['patience'], 
            restore_best_weights=True
        )
        
        # If the datasets are empty, cannot train
        if datasets['validation']['ds'] is None or datasets['training']['ds'] is None:
            print(colored(
                f"Non-fatal Error: training was skipped for the test subject {testing_subject} and val subject {validation_subject}. " + 
                f"There were no files in the training and/or validation datasets.\n",
                'yellow'
            ))
            continue
        
        # Train the model
        time_start = perf_counter()
        history = model.fit(
            datasets['training']['ds'],
            validation_data=datasets['validation']['ds'],
            epochs=config['hyperparameters']['epochs'],
            callbacks=[early_stopping_callbacks]
        )
        time_elapsed = perf_counter() - time_start
        
        # Output the results
        print(colored(f"Finished training for testing subject {testing_subject} and validation subject {validation_subject}.", 'green'))
        output_results(config['output_path'], testing_subject, validation_subject, rotation, model, model_type, history, time_elapsed, datasets, config['class_names'])
        

""" Main functions -------------------------------------------- """
def main():
    """ Runs the training process for each configuration and test subject. """
    
    # Print a list of the available GPUs
    show_gpu_list()
    
    # Parse the command line arguments
    configs = parse_training_configs('./training/training_config_files')
    for config in configs:
        
        # Make sure the subject list is of the same case
        config['subject_list'] = [s.lower() for s in config['subject_list']]
        
        # Train for each test subject
        for test_subject in config['subject_list']:
            print(colored(
                f"\n\n===========================================================\n" + 
                f"Starting training for {test_subject} in {config['selected_model_name']}\n"
                , 'magenta'
            ))
            files, folds, rotations, indexes, label_position = training_prep(config, test_subject)
            training_loop(config, test_subject, files, folds, rotations, indexes, label_position)
    

if __name__ == "__main__":
    """ Called when this file is run. """
    main()
