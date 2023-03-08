import json
import random

import numpy as np
import os, sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from time import perf_counter
from tensorflow import keras
import re
import glob
import pandas as pd
import argparse
import csv
from create_model import *


def create_parser():
    """
        Create parser.

        :return: List of names of all image files
    """
    parser = argparse.ArgumentParser(description='Medical Image', fromfile_prefix_chars='@')
    # High-level commands
    parser.add_argument('--config_file', '-config_file', type=str, default=None, help='Path to a config file')
    parser.add_argument('--config_folder', '-config_folder', type=str, default='./Default_config_files', help="Path to a config files folder(not trailing '/')")
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
            folds.append({'train': [], 'test': []})
            folds[-1]['test'].append(item_test)
            for j, item_train in enumerate(subject_list):
                # j is the test subject
                if i != j:
                    folds[-1]['train'].append(item_train)
    print("-----------------folds:")
    print(folds)
    print("-----------------")
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
        print("folder: %s, img count: %d"%(item, img_count))
    print("folder count: %d"%(dir_count))
    print("image count: %d"%(img_count_total))
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
    print("\033[32m%d config file found!\033[0m"% file_count)
    print(configFile_list)
    return configFile_list


# Get labels, subject, class index
def get_label_subject(path,label_position ,class_names, subject_list):
    #"/home/xx_fat_xxx/img_ps/ep_optic_axis/train/E4_1_fat_3_Optic%20axis/807_E4_1_fat_3_Optic%20axis.png"
    formatted_path=path.replace("%20", " ")
    # Remove the file extension
    temp = formatted_path.split('.')
    fileName_remove_extension = temp[0]
    if len(temp) > 2:
        for part in parts_0[1:-1]:
            fileName_remove_extension = fileName_remove_extension + part
    # fileName_remove_extension split by "_"
    fileName_parts = fileName_remove_extension.split('_')

    # Get all match the labels
    labels = [class_name for class_name in class_names if class_name in fileName_parts]

    # Assuming only 1 label will be obtained, otherwise throw exception
    if len(labels) < 1:
        raise Exception("Error when getting label from: " + path)
    if len(set(labels)) > 1:
        raise Exception("Duplicate labels extracted from: " + path)

    # Update label index within the filename, this line only perform once
    if label_position==-1:
        label_position=fileName_parts.index(labels[0])

    # parts_0 = tf.strings.split(filename, ".")
    # complete = parts_0[0]
    # if len(parts_0) > 2:
    #     for part in parts_0[1:-1]:
    #         complete = tf.add(complete, part)

    idx=class_names.index(labels[0])
    # Get all match the subjects
    subjects = [subject for subject in subject_list if subject in fileName_parts]


    # Assuming only 1 subject will be obtained, otherwise throw exception
    if len(set(subjects)) > 1:
        raise Exception("Duplicate subjects extracted from: " + path)
    elif len(subjects) < 1:
        raise Exception("Error when getting subject from: " + path)

    return labels[0], idx, subjects[0], label_position


def parse_image(filename, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width):

    parts_0 = tf.strings.split(filename, ".")
    complete = parts_0[0]
    if len(parts_0) > 2:
        for part in parts_0[1:-1]:
            complete = tf.add(complete,part)


    parts = tf.strings.split(complete, "_")
    label = parts[label_position]
    # tf.print(label)
    # tf.print("length:"+str(tf.shape(parts)))
    label_bool = (label == class_names)
    # To double check the file name and label is correct

    # tf.print(filename, label, output_stream=sys.stderr, sep=',')

    image = tf.io.read_file(filename)
    # image = tf.image.decode_jpeg(image, channels=channels)
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = tf.io.decode_image(image, channels=channels, dtype=tf.float32,
                               name=None, expand_animations=False)
    if do_cropping == 'true':
       image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
    if use_mean == 'true':
        image = image - mean / 255

    return image, tf.argmax(label_bool)






def training(data, testing_subject, config_name):
    # Get batch size from config
    batch_size = int(data['batch_size'])  # (int)
    # Get epochs number from config
    n_epochs = int(data['epochs'])  # (int)
    # Get subject list from config
    subject_list = data['subject_list'].split(',')  # (string list)
    # subject_list = [x for x in subject_list]  # (string list lower case)
    # Get the main directory path from config
    file_path = data['files_directory']  # (string)
    # Get the saving directory path from config
    results_path = data['results_path']  # (string)
    # Get the seed from config
    the_seed = int(data['seed'])  # (int)
    # Get the classes name from config
    class_names = data['classes_names'].split(',')  # (string list)
    # learning_rate
    learning_rate = float(data['learning_rate'])  # (float)
    # momentum
    the_momentum = float(data['momentum'])  # (float)
    # decay
    the_decay = float(data['decay'])  # (float)
    # patience
    the_patience = int(data['patience'])  # (int)
    # channels
    channels = int(data['channels'])  # (int)
    # Get the mean
    mean = float(data['mean'])  # (int)
    # Shuffle the folds
    shuffle_the_folds = data['shuffle_the_folds']  # (string)
    # Get use_mean
    use_mean = data['use_mean']  # (string)
    # Get cropping position
    cropping_position = data['cropping_position'].split(',')  # (string list)
    # Get image size
    image_size = data['image_size'].split(",")  # (string list)
    offset_height = int(cropping_position[0])  # (int)
    offset_width = int(cropping_position[1])  # (int)
    target_height = int(data['target_height'])  # (int)
    target_width = int(data['target_width'])  # (int)
    # Get cropping flag
    do_cropping = data['do_cropping']  # (string)
    # Get model type
    selected_model = data['selected_model']  # (string)
    # # Get rotations from config
    # rotations_config = data['rotations']  # (string)
    # rotations = 0  # (int)
    label_position = -1  # (int)

    # Create result folder if not existed
    if not os.path.exists("%s" % (results_path)):
        os.makedirs("%s" % (results_path))
    if not os.path.exists("%s/Test_subject_%s" % (results_path, testing_subject)):
        os.makedirs("%s/Test_subject_%s" % (results_path, testing_subject))
    if not os.path.exists("%s/Test_subject_%s/%s" % (results_path, testing_subject, config_name)):
        os.makedirs("%s/Test_subject_%s/%s" % (results_path, testing_subject, config_name))
    results_path = results_path + '/Test_subject_' + testing_subject + '/' + config_name

    # [2][3,4,5,6,7,8]
    # [3][2,4,5,6,7,8]
    # [4][2,3,5,6,7,8]
    # Generate folds based on subject list
    folds = split_folds(subject_list, testing_subject)
    # Shuffle the folds if needed
    if shuffle_the_folds == "true":
        random.shuffle(folds)

    # Get the directories with the full directory name
    filename_list_all = get_filename_list(file_path)

    # Shuffle the file list
    # Set the seed
    random.seed(the_seed)
    # Sort
    filename_list_all.sort()
    # Shuffle
    random.shuffle(filename_list_all)
    label_list_all=[]
    label_index_list_all=[]
    subject_list_all=[]


    for file_name in filename_list_all:
        label, idx, subject,label_position =get_label_subject(file_name,label_position, class_names, subject_list)
        label_list_all.append(label)
        label_index_list_all.append(idx)
        subject_list_all.append(subject)

    print("Unique labels numbers:" + str(len(set(label_list_all))))
    print("Unique labels:" + str(set(label_list_all)))





    # Before training, verify the parameters
    print("====================================")
    print("\033[91mCommon parameters\033[0m")
    print("file total length: " + str(len(filename_list_all)))
    print(str(n_epochs) + " epochs for each training, with batch size " + str(batch_size))
    print("Cropping position: " + str(offset_height) + "," + str(offset_width) + "; Image size: " + str(target_height) + "," + str(target_width) + "; Channel: " + str(channels))
    print("Label position in file name split by '_': " + str(label_position))
    print("Subjects list: " , end =" ")
    print(*subject_list, sep=', ')
    print("Classes list: " , end =" ")
    print(*class_names, sep=', ')
    print("subject list length:" +  str(len(set(subject_list_all))))
    print(subject_list_all[0])
    print("====================================")


    train_file_name_list=[]
    train_label_list = []
    train_index_list = []
    test_file_name_list = []
    test_label_list = []
    test_index_list = []

    # val_subject=folds[rot]['val'][0]
    test_subject = folds[0]['test'][0]
    for idx, file in enumerate(filename_list_all):
        if subject_list_all[idx] == folds[0]['test'][0]:
            test_file_name_list.append(file)
            test_label_list.append(label_list_all[idx])
            test_index_list.append(label_index_list_all[idx])
        else:
            train_file_name_list.append(file)
            train_label_list.append(label_list_all[idx])
            train_index_list.append(label_index_list_all[idx])
    print("------------------------------------")
    print("\033[91mStatus of Rotation\033[0m: 0" )
    print("Test with:", end =" ")
    print(*folds[0]['test'], sep=', ')
    # print("Train with:", end =" ")
    # print(*folds[0]['val'], sep=', ')
    print("Train with:", end =" ")
    print(*folds[0]['train'], sep=', ')
    print("Length of train files: " + str(len(train_file_name_list)))
    # print("Length of val files: " + str(len(val_file_name_list)))
    print("Length of test files: " + str(len(test_file_name_list)))
    print("------------------------------------")


    list_train_ds = tf.data.Dataset.from_tensor_slices(train_file_name_list)
    # list_val_ds = tf.data.Dataset.from_tensor_slices(val_file_name_list)
    files_test_ds = tf.data.Dataset.from_tensor_slices(test_file_name_list)

    images_train_ds_v2 = list_train_ds.map(lambda x:parse_image(x, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width))
    images_train_batch_ds = images_train_ds_v2.batch(batch_size, drop_remainder=False)

    image_test_ds_v2 = files_test_ds.map(lambda x: parse_image(x, mean, use_mean, class_names, label_position, channels, do_cropping, offset_height, offset_width, target_height, target_width))
    images_test_batch_ds = image_test_ds_v2.batch(batch_size, drop_remainder=False)

    base_model_empty, model_type = get_model(selected_model,target_height, target_width, channels)
    print("\033[91mCurrent model: %s\033[0m" % model_type)

    avg = keras.layers.GlobalAveragePooling2D()(base_model_empty.output)
    out_put = keras.layers.Dense(len(class_names), activation="softmax")(avg)
    model_ready = keras.models.Model(inputs=base_model_empty.input, outputs=out_put)

    optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=the_momentum, nesterov=True, decay=the_decay)


    model_ready.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                           metrics=["accuracy"])

    t1_start = perf_counter()

    history = model_ready.fit(images_train_batch_ds,
                                 batch_size=batch_size,
                                 validation_data=None,
                                 epochs=n_epochs
                                 )

    t1_stop = perf_counter()
    time_lapse = t1_stop - t1_start
    print("fold 0 Time lapse:" , time_lapse)

    if not os.path.exists("%s/%s_test_%s" % (results_path, model_type, test_subject)):
        os.makedirs("%s/%s_test_%s" % (results_path, model_type, test_subject))
    if not os.path.exists("%s/%s_test_%s/model" % (results_path, model_type, test_subject)):
        os.makedirs("%s/%s_test_%s/model" % (results_path, model_type, test_subject))
    if not os.path.exists("%s/%s_test_%s/true_label" % (results_path, model_type, test_subject)):
        os.makedirs("%s/%s_test_%s/true_label" % (results_path, model_type, test_subject))
    if not os.path.exists("%s/%s_test_%s/file_name" % (results_path, model_type, test_subject)):
        os.makedirs("%s/%s_test_%s/file_name" % (results_path, model_type, test_subject))
    if not os.path.exists("%s/%s_test_%s/prediction" % (results_path, model_type, test_subject)):
        os.makedirs("%s/%s_test_%s/prediction" % (results_path, model_type, test_subject))




    rot = 0
    print("Saving model for fold 0")
    model_ready.save("%s/%s_test_%s/model/%s_test_%s_resnet50.h5" % (results_path, model_type, test_subject, model_type, test_subject))

    print("Elapsed time during the whole program in seconds for fold: %s (test_%s): " % (rot, test_subject), time_lapse)

    print("Saving timing file for fold 0")
    with open('%s/%s_test_%s/%s_test_%s_time-total.csv' % (results_path, model_type, test_subject, model_type, test_subject), 'w') as f:
        writer = csv.writer(f)
        writer.writerow([time_lapse])
        # np.save(f, np.array(time_lapse))

    print("Saving classes names file.")
    with open('%s/%s_test_%s/%s_test_%s_classes-names.csv' % (results_path, model_type, test_subject, model_type, test_subject), 'w') as f:
        writer = csv.writer(f)
        for idx, item in enumerate(class_names):
            writer.writerow(item)
            
    test_pred = model_ready.predict(images_test_batch_ds)
    print("Saving prediction of test for fold 0")
    with open('%s/%s_test_%s/prediction/%s_test_%s_test-predicted.csv' % (results_path, model_type, test_subject, model_type, test_subject), 'w') as f:
        writer = csv.writer(f)
        for item in test_pred:
            writer.writerow(item)
        # np.save(f, test_pred)

    test_eva = model_ready.evaluate(images_test_batch_ds)
    print("Saving evaluation of test for fold 0")
    with open('%s/%s_test_%s/%s_test_%s_evaluation_test.csv' % (results_path, model_type, test_subject, model_type, test_subject), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(test_eva)
    pd_history_history = pd.DataFrame.from_dict(history.history)
    pd_history_history.to_csv('%s/%s_test_%s/%s_test_%s_history.csv' % (results_path, model_type, test_subject, model_type, test_subject))


    with open('%s/%s_test_%s/true_label/%s_test_%s_test-true-label.csv' % (results_path, model_type, test_subject, model_type, test_subject), 'w') as f:
        writer = csv.writer(f)
        for item in test_label_list:
            writer.writerow([item])

    with open('%s/%s_test_%s/file_name/%s_test_%s_test-file.csv' % (results_path, model_type, test_subject, model_type, test_subject), 'w') as f:
        writer = csv.writer(f)
        for item in test_file_name_list:
            writer.writerow([item])

    with open('%s/%s_test_%s/true_label/%s_test_%s_test-true-label-index.csv' % (results_path, model_type, test_subject, model_type, test_subject), 'w') as f:
        writer = csv.writer(f)
        for item in test_index_list:
            writer.writerow([item])






# format of dir
# Get available gpu list
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

parser = create_parser()
args = parser.parse_args()
configFile_list = []

if args.config_file is None:
    print("\033[91mNo config file specified, checking config folder...\033[0m")
    if os.path.isdir(args.config_folder):
        configFile_list = get_configFile_list(args.config_folder)
    else:
        raise Exception("Please specify correct config folder")
else:
    configFile_list.append(args.config_file)

f = open(configFile_list[0])
test_config =json.load(f)
f.close()
testing_subjects = test_config['testing_subject'].split(',')  # (string list)
testing_subjects = [x for x in testing_subjects]  # (string list lower case)



for testing_subject in testing_subjects:
    for configFile_path in configFile_list:
        # Read configurations from config.json
        f = open(configFile_path)
        configurations = json.load(f)
        f.close()
        config_name = configFile_path.split("/")[-1].split(".")[0]
        training(configurations, testing_subject, config_name)







