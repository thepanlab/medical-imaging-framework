from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from image_parser import parse_image
from time import perf_counter
from termcolor import colored
from tensorflow import keras
from create_model import *
import checkpoint as ckpt
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os, sys
import random
import json
import glob
import csv
import re


def create_parser():
    """
        Create parser.

        :return: List of names of all image files
    """
    parser = argparse.ArgumentParser(description='Medical Image', fromfile_prefix_chars='@')
    # High-level commands
    parser.add_argument('--config_file', '-config_file', '-c', type=str, default=None, help='Path to a config file')
    parser.add_argument('--config_folder', '-config_folder', '-f', type=str, default='./config_files', help="Path to a config files folder(not trailing '/')")
    return parser


def split_folds(subject_list, testing_subject):
    """
        Split folds based on subject_list.
        Just for inner loop, after selecting test subject and config file.
        There's an option in config file "rotations",
        "rotations": "all" means perform all the combinations,
        or a number indicates how many rotations to perform for each of the test subject,
        e.g.: "rotations": "3"

        output e.g.: if there are 4 subjects, the test subject is "e1", the output would be:
        [
            {e1}, {e2}, {...}
            {e1}, {e3}, {...}
            {e1}, {e4}, {...}
        ]
        :return: List of all folds combinations
    """
    folds = []
    for i, item_test in enumerate(subject_list):
        # i is the test subject
        if item_test == testing_subject:
            for j, item_val in enumerate(subject_list):
                # j is the test subject
                if i != j:
                    folds.append({'train': [], 'val': [], 'test': []})
                    folds[-1]['test'].append(item_test)
                    folds[-1]['val'].append(item_val)
                    for k, item_train in enumerate(subject_list):
                        # i is the training subject
                        if (i != k) and (j != k):
                            folds[-1]['train'].append(item_train)
    print(colored(f"-----------------\nFolds:\n{folds}\n-----------------"))
    return folds


# def get_subject(file_address, letter_search):
#     x = re.search(letter_search + "[0-9]+", get_filename(file_address) )
#     return x.group(0)


def get_filename_list(path):
    """
        Get the directories with the full directory name and print dirs count and img count.

        :path path: path to the file dir
        :return: List of names of all image files
    """
    dirs = os.listdir(path)
    dir_count = 0
    img_count_total = 0
    file_list = []
    for item in dirs:
        second_dir = path + "/" + item
        dir_count += 1
        img_count = 0
        if os.path.isdir(second_dir):
            sub_dirs = os.listdir(second_dir)
            for sub_item in sub_dirs:
                third_dir = second_dir + "/" + sub_item
                if os.path.isdir(third_dir):
                    third_dir_files = os.listdir(third_dir)
                    for third_dir_file in third_dir_files:
                        final_path = third_dir + "/" + third_dir_file
                        img_count += 1
                        img_count_total += 1
                        file_list.append(final_path)
                else:    
                    img_count += 1
                    img_count_total += 1
                    file_list.append(third_dir)
        print(colored(f"Folder: {item}, Image count: {img_count}", 'green'))
    print(colored(f"Folder count: {dir_count}, Image count: {img_count_total}", 'green'))
    return file_list


def get_configFile_list(path):
    file_count = 0
    configFile_list = []
    files = os.listdir(path)
    for item in files:
        complete_path = path + "/" + item
        if not os.path.isdir(complete_path):
            file_count += 1
            configFile_list.append(complete_path)
    print(colored(f"Config file found! {file_count}", "green"))
    print(colored(configFile_list, "yellow"))
    return configFile_list


# Get labels, subject, class index
def get_label_subject(path,label_position ,class_names, subject_list):
    #"/home/xx_fat_xxx/img_ps/ep_optic_axis/train/E4_1_fat_3_Optic%20axis/807_E4_1_fat_3_Optic%20axis.png"
    formatted_path=path.lower().replace("%20", " ")
    # Get all match the labels
    
    # Assuming only 1 label will be obtained, otherwise throw exception
    if len(labels) > 1:
        raise Exception(colored(f"Error: Duplicate labels extracted from: {path}", 'red'))
    elif len(labels) < 1:
        raise Exception(colored(f"Error: Could not get label from: {path}", 'red'))

    # Update label index within the filename, this line only perform once
    if label_position==-1:
        temp = os.path.abspath(formatted_path).split('.')
        label_position=temp[0].split('_').index(labels[0])

    # parts_0 = tf.strings.split(filename, ".")
    # complete = parts_0[0]
    # if len(parts_0) > 2:
    #     for part in parts_0[1:-1]:
    #         complete = tf.add(complete, part)

    idx=class_names.index(labels[0])
    # Get all match the subjects
    subjects = [subject for subject in subject_list if subject in formatted_path]
    # Assuming only 1 subject will be obtained, otherwise throw exception
    if len(subjects) > 1:
        raise Exception(colored(f"Duplicate subjects extracted from: {path}"))
    elif len(subjects) < 1:
        raise Exception(colored(f"Error when getting subject from: {path}"))

    return labels[0], idx, subjects[0], label_position



def training(data, testing_subject, config_name):
    # Get configuration
    batch_size = int(data['batch_size'])  # Batch size (int)
    n_epochs = int(data['epochs'])        # Num epochs (int)
    subject_list = [x.lower() for x in data['subject_list'].split(',')]  # subjects (string list lower case)
    file_path = data['files_directory']   # main directory path (string)
    results_path = data['results_path']   # Saving directory (string)
    the_seed = int(data['seed'])          # Seed (int)
    class_names = data['classes_names'].split(',')  # Class names (string list)
    learning_rate = float(data['learning_rate'])    # Learning rate (float)
    the_momentum = float(data['momentum'])  # Momentum (float)
    the_decay = float(data['decay'])        # Decay (float)
    the_patience = int(data['patience'])    # Patience (int)
    channels = int(data['channels'])    # Channels (int)
    mean = float(data['mean'])          # Mean (int)
    use_mean = data['use_mean']         # Use mean (string)
    cropping_position = data['cropping_position'].split(',')  # Cropping pos (string list)
    image_size = data['image_size'].split(",")  # Image size (string list)
    offset_height = int(cropping_position[0])   # Offset height (int)
    offset_width = int(cropping_position[1])    # Offset width (int)
    target_height = int(data['target_height'])  # Target height (int)
    target_width = int(data['target_width'])    # Target width (int)
    do_cropping = data['do_cropping']           # Cropping flag (string)
    selected_model = data['selected_model']     # Selected model (string)
    rotations_config = data['rotations']        # Rotations (string)

    # Create result folders if they don't exist
    folder = f"{results_path}/Test_subject_{testing_subject}/{config_name}"
    folder_split = list(filter(None, folder.split('/')))
    for folder_index in range(len(folder_split)):
        path_part = folder_split[0]
        for i in range(1, folder_index): path_part += f'/{folder_split[i]}'
        if not os.path.exists(path_part): os.makedirs(path_part)

    # Generate folds based on subject list and get rotations
    rotations = 0
    folds = split_folds(subject_list, testing_subject)
    if rotations_config == 'all' or int(rotations_config) >= len(folds):
        rotations = len(folds)
    # elif int(rotations_config) >= len(folds):
    #     rotations = len(folds)
    else:
        rotations = int(rotations_config)

    # Get the directories with the full directory name
    filename_list_all = get_filename_list(file_path)

    # Shuffle the file list
    random.seed(the_seed)               # Set seed
    filename_list_all.sort()            # Sort
    random.shuffle(filename_list_all)   # Shuffle
    label_list_all=[]
    label_index_list_all=[]
    subject_list_all=[]

    label_position = -1
    for file_name in filename_list_all:
        label, idx, subject, label_position = get_label_subject(file_name, label_position, class_names, subject_list)
        label_list_all.append(label)
        label_index_list_all.append(idx)
        subject_list_all.append(subject)

    # Before training, verify the parameters
    print(colored(
        f"====================================\nCommon parameters:" +
        f"\nTotal files: {len(filename_list_all)} \nTotal rotations: {rotations}" +
        f"\n{n_epochs} epochs for each training, with batch size {batch_size}" + 
        f"\nCropping position: ({offset_height}, {offset_width}); Image size: ({target_height}, {target_width}); Channels: {channels}" +
        f"\nLabel position in file name split by '_': {label_position}" + 
        f"\nSubject list length: {len(set(subject_list_all))}\n{subject_list_all[0]}" +
        f"\n\nSubjects list:", 'blue'))
    print(colored([*subject_list], 'yellow'))
    print(colored(f"\nClasses list:", 'blue'))
    print(colored([*class_names], 'yellow'))
    print(colored(f"\n====================================", 'blue'))


    for rot in range(rotations):
        train_file_name_list=[]
        train_label_list = []
        train_index_list = []
        val_file_name_list = []
        val_label_list = []
        val_index_list = []
        test_file_name_list = []
        test_label_list = []
        test_index_list = []

        val_subject=folds[rot]['val'][0]
        test_subject = folds[rot]['test'][0]
        for idx, file in enumerate(filename_list_all):
            if subject_list_all[idx] == folds[rot]['val'][0]:
                val_file_name_list.append(file)
                val_label_list.append(label_list_all[idx])
                val_index_list.append(label_index_list_all[idx])
            elif subject_list_all[idx] == folds[rot]['test'][0]:
                test_file_name_list.append(file)
                test_label_list.append(label_list_all[idx])
                test_index_list.append(label_index_list_all[idx])
            else:
                train_file_name_list.append(file)
                train_label_list.append(label_list_all[idx])
                train_index_list.append(label_index_list_all[idx])

        print(colored(
            "------------------------------------\n" +
            f"Status of Rotation: {rot}", "blue"))

        print(colored(f"\nTest with: ", 'blue'))
        print(colored([*folds[rot]['test']], 'blue'), sep=', ')

        print(colored(f"\nValidation with: ", 'blue'))
        print(colored([*folds[rot]['val']], 'blue'), sep=', ')

        print(colored(f"\nTrain with: ," 'blue'))
        print(colored([*folds[rot]['train']], 'blue'), sep=', ')

        print(colored(
            f"\nAmount of training files: {len(train_file_name_list)}" +
            f"\nAmount of validation files: {len(val_file_name_list)}" +
            f"\nAmount of testing files: {len(test_file_name_list)}" +
            "\n------------------------------------", "blue"))

        list_train_ds = tf.data.Dataset.from_tensor_slices(train_file_name_list)
        list_val_ds = tf.data.Dataset.from_tensor_slices(val_file_name_list)
        files_test_ds = tf.data.Dataset.from_tensor_slices(test_file_name_list)

        images_train_ds_v2 = list_train_ds.map(lambda x:parse_image(x, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width))
        images_train_batch_ds = images_train_ds_v2.batch(batch_size, drop_remainder=False)

        images_val_ds_v2 = list_val_ds.map(lambda x:parse_image(x, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width))
        images_val_batch_ds = images_val_ds_v2.batch(batch_size, drop_remainder=False)

        image_test_ds_v2 = files_test_ds.map(lambda x: parse_image(x, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width))
        images_test_batch_ds = image_test_ds_v2.batch(batch_size, drop_remainder=False)

        base_model_empty, model_type = get_model(selected_model,target_height, target_width, channels)
        print(colored(f"Current model: {model_type}", 'green'))
        #
        # base_model_empty = keras.applications.resnet50.ResNet50(include_top=False,
        #                                                         weights=None,
        #                                                         input_tensor=None,
        #                                                         input_shape=(target_height, target_width, channels),
        #                                                         pooling=None)


        avg = keras.layers.GlobalAveragePooling2D()(base_model_empty.output)
        out_put = keras.layers.Dense(len(class_names), activation="softmax")(avg)
        model_ready = keras.models.Model(inputs=base_model_empty.input, outputs=out_put)

        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=the_momentum, nesterov=True, decay=the_decay)

        early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=the_patience,
                                                          restore_best_weights=True)


        model_ready.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                               metrics=["accuracy"])

        t1_start = perf_counter()


        # Create a method of checkpointing
        file_prefix = f"{model_type}_{rot}_test_{test_subject}_val_{val_subject}"
        checkpoints_path = f"{data['checkpoints_path']}/Test_subject_{testing_subject}/{config_name}"
        checkpoints = ckpt.create_checkpoint(checkpoints_path, file_prefix + '.hdf5')

        # Fit data
        history = model_ready.fit(images_train_batch_ds,
                                     batch_size=batch_size,
                                     validation_data=images_val_batch_ds,
                                     epochs=n_epochs,
                                     callbacks=[early_stopping_cb, checkpoints])

        t1_stop = perf_counter()
        time_lapse = t1_stop - t1_start
        print(colored(f"Fold {rot}'s time lapse:", "green"))

        # Create output subfolders if not exist
        path_prefix = f"{results_path}/{file_prefix}"
        subfolders = ['', '/model', '/true_label', '/file_name', '/prediction']
        for folder in subfolders:
            folder_path = path_prefix + folder
            if not os.path.exists(folder_path): os.makedirs(folder_path)

        print(colored(f"Saving model for fold: {rot} ", "green"))
        model_ready.save(f"{path_prefix}/model/{file_prefix}_resnet50.h5")

        print(colored(f"Elapsed time during the whole program in seconds for fold: {rot} (test_{test_subject}_val_{val_subject}): {time_lapse}", 'yellow'))
        print(colored(f"Saving timing file for fold: {rot} ", "green"))
        with open(f"{path_prefix}/{file_prefix}_time-total.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow([time_lapse])
            # np.save(f, np.array(time_lapse))

        val_pred = model_ready.predict(images_val_batch_ds)
        print(colored(f"Saving prediction of validation for fold: {rot} ", "green"))
        with open(f"{path_prefix}/prediction/{file_prefix}_val_predicted.csv", 'w') as f:
            writer = csv.writer(f)
            for item in val_pred:
                writer.writerow(item)
            # np.save(f, val_pred)

        test_pred = model_ready.predict(images_test_batch_ds)
        print(colored(f"Saving prediction of test for fold: {rot} ", "green"))
        with open(f"{path_prefix}/prediction/{file_prefix}_test_predicted.csv", 'w') as f:
            writer = csv.writer(f)
            for item in test_pred:
                writer.writerow(item)
            # np.save(f, test_pred)

        test_eva = model_ready.evaluate(images_test_batch_ds)
        print(colored(f"Saving evaluation of test for fold: {rot} ", "green"))
        with open(f"{path_prefix}/{file_prefix}_evaluation_test.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(test_eva)
            # np.save(f, test_eva)

        pd_history_history = pd.DataFrame.from_dict(history.history)
        pd_history_history.to_csv(f"{path_prefix}/{file_prefix}_history.csv")

        with open(f"{path_prefix}/true_label/{file_prefix}_val_true_label.csv", 'w') as f:
            writer = csv.writer(f)
            for item in val_label_list:
                writer.writerow([item])
            # np.save(f, val_label_list)

        with open(f'{path_prefix}/true_label/{file_prefix}_test_true_label.csv', 'w') as f:
            writer = csv.writer(f)
            for item in test_label_list:
                writer.writerow([item])
            # np.save(f, test_label_list)

        with open(f'{path_prefix}/file_name/{file_prefix}_val_file.csv', 'w') as f:
            writer = csv.writer(f)
            for item in val_file_name_list:
                writer.writerow([item])
            # np.save(f, val_file_name_list)

        with open(f'{path_prefix}/file_name/{file_prefix}_test_file.csv', 'w') as f:
            writer = csv.writer(f)
            for item in test_file_name_list:
                writer.writerow([item])
            # np.save(f, test_file_name_list)

        with open(f'{path_prefix}/true_label/{file_prefix}_test_true_label_index.csv', 'w') as f:
            writer = csv.writer(f)
            for item in test_index_list:
                writer.writerow([item])
            # np.save(f, test_index_list)

        with open(f'{path_prefix}/true_label/{file_prefix}_val_true_label_index.csv', 'w') as f:
            writer = csv.writer(f)
            for item in val_index_list:
                writer.writerow([item])
            # np.save(f, val_index_list)


if __name__ == "__main__":
    """ The Main Program """
    # Get available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print(colored(f"Name: {gpu.name}, Type: {gpu.device_type}", "green"))
    # os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # Parse configuration from command line
    parser = create_parser()
    args = parser.parse_args()
    configFile_list = []
    if args.config_file is None:
        print(colored(f"Warning: No config file specified, checking config folder...", 'red'))
        if os.path.isdir(args.config_folder):
            configFile_list = get_configFile_list(args.config_folder)
        else:
            raise Exception(colored(f"Error: Please specify correct config folder instead of '{args.config_folder}'", 'red'))
    else:
        configFile_list.append(args.config_file)

    # Load the configuration
    with open(configFile_list[0]) as fp:
        test_config =json.load(fp)

    # Get the subjects as a list of lowercase strings
    testing_subjects = [x.lower() for x in test_config['testing_subject'].split(',')]

    # For each subject, open its configuration and train
    for testing_subject in testing_subjects:
        for configFile_path in configFile_list:

            # Read configurations from config.json
            with open(configFile_path) as fp:
                configurations = json.load(fp)
            config_name = configFile_path.split("/")[-1].split(".")[0]
            training(configurations, testing_subject, config_name)



"""
test_e1:
        config_1:
                resnet_50_1_test_e1


"""


"""

test_e1
        -config_1
                -val_e2
                -val_e3
        -config_2
test_e2
        -config_1
        -config_2
test_e3
        -config_1
        -config_2

dir structure:

config_1.json
e1_results_path - config_1 -
                - config_2 -
                - config_3 -   * model_1_test_e1_val_e2 - ...
                             - * model_1_test_e1_val_e3 - ...
                             - * model_1_test_e1_val_e4 - * model - model_1_test_e1_val_e4_resnet50.h5
                                       ...              - * true_label - model_1_test_e1_val_e4_val_true-label.npy # to csv
                                                                       - model_1_test_e1_val_e4_test_true-label.npy # to csv
                                                                       - model_1_test_e1_val_e4_val_true-label-index.npy # to csv
                                                                       - model_1_test_e1_val_e4_test_true-label-index.npy # to csv                                               
                                                        - * file_name - model_1_test_e1_val_e4_val_file.npy # to csv
                                                                      - model_1_test_e1_val_e4_test_file.npy # to csv    
                                                        - * prediction - model_1_test_e1_val_e4_val_predicted.npy # to csv
                                                                       - model_1_test_e1_val_e4_test_predicted.npy # to csv                             
                                                        - model_1_test_e1_val_e4_history.csv
                                                        - model_1_test_e1_val_e4_evaluation_test.npy # to csv
                            - * model_1_test_e3_val_e1
                            
                    
----                    
config_2.json
config_2_results_path - * model_2_test_e1_val_e2 - ...
                     - * model_2_test_e1_val_e3 - ...
                     - * model_2_test_e1_val_e4 - * model - model_2_test_e1_val_e4_resnet50.h5
                               ...              - * true label - model_2_test_e1_val_e4_val_true-label.npy # to csv
                                                               - model_2_test_e1_val_e4_test true label.npy # to csv
                                                               - model_2_test_e1_val_e4_val true label index.npy # to csv
                                                               - model_2_test_e1_val_e4_test true label index.npy # to csv                                               
                                                - * file name - model_2_test_e1_val_e4_val file.npy # to csv
                                                              - model_2_test_e1_val_e4_test file.npy # to csv    
                                                - * prediction - model_2_test_e1_val_e4_val predicted.npy # to csv
                                                               - model_2_test_e1_val_e4_test predicted.npy # to csv                             
                                                - model_2_test_e1_val_e4_history.csv
                                                - model_2_test_e1_val_e4_evaluation_test.npy # to csv
                            .....(30 results for 6 subjects)
                     - * model_2_test_e3_val_e1
                     - * model_2_test_e3_val_e2
----                     
config_3.json
config_3_results_path - * model_3_test_e1_val_e2 - ...
                     - * model_3_test_e1_val_e3 - ...
                     - * model_3_test_e1_val_e4 - * model - model_2_test_e1_val_e4_resnet50.h5
                               ...              - * true label - model_3_test_e1_val_e4_val_true-label.npy # to csv
                                                               - model_3_test_e1_val_e4_test true label.npy # to csv
                                                               - model_3_test_e1_val_e4_val true label index.npy # to csv
                                                               - model_3_test_e1_val_e4_test true label index.npy # to csv                                               
                                                - * file name - model_3_test_e1_val_e4_val file.npy # to csv
                                                              - model_3_test_e1_val_e4_test file.npy # to csv    
                                                - * prediction - model_3_test_e1_val_e4_val predicted.npy # to csv
                                                               - model_3_test_e1_val_e4_test predicted.npy # to csv                             
                                                - model_3_test_e1_val_e4_history.csv
                                                - model_3_test_e1_val_e4_evaluation_test.npy # to csv
                            .....(30 results for 6 subjects)
                     - * model_3_test_e3_val_e1
                     - * model_3_test_e3_val_e2

"""






