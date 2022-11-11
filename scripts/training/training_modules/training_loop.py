from training.training_modules.result_outputter import output_results
from training.training_modules.model_creator import TrainingModel
from training.training_modules.image_parser import parse_image
from termcolor import colored
from time import perf_counter
from tensorflow import keras
import tensorflow as tf


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
    
    # rotation --> n_folds TODO
    
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
            
            # TODO this is where images are parsed. Change for 3D data?
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
        model_obj = TrainingModel(
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
        history = model_obj.model.fit(
            datasets['training']['ds'],
            validation_data=datasets['validation']['ds'],
            epochs=config['hyperparameters']['epochs'],
            callbacks=[early_stopping_callbacks]
        )
        time_elapsed = perf_counter() - time_start
        
        # Output the results
        print(colored(f"Finished training for testing subject {testing_subject} and validation subject {validation_subject}.", 'green'))
        output_results(config['output_path'], testing_subject, validation_subject, rotation, model_obj, history, time_elapsed, datasets, config['class_names'])
        
