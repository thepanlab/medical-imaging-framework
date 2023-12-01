import random
from collections import namedtuple
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from util.get_config import parse_json

# https://www.geeksforgeeks.org/reading-and-writing-json-to-a-file-in-python/#

def get_array_lr(config):
    
    lr_min = config["hyperparameters"]['learning_rate_min']
    lr_max = config["hyperparameters"]['learning_rate_max']
    
    log_lr_min = np.log10(lr_min)
    log_lr_max = np.log10(lr_max)
    
    a_log_lr = np.arange(log_lr_min, log_lr_max+1, 1)
    a_lr = 10**a_log_lr
    
    return a_lr


def get_array_batch(config):
    
    batch_size_min = config["hyperparameters"]["batch_size_min"]
    batch_size_max = config["hyperparameters"]["batch_size_max"]
    
    log2_batch_size_min  = np.log2(batch_size_min)
    log2_batch_size_max = np.log2(batch_size_max)
    
    a_log2_batch_size = np.arange(log2_batch_size_min,
                                  log2_batch_size_max+1, 1)
    
    a_batch_size = 2**a_log2_batch_size
    
    return a_batch_size
    

def create_dict_json(temp_variable, config):

    dict_json = {}
    
    dict_hyperparameters = {}

    # random values    
    dict_hyperparameters["batch_size"] = int(temp_variable.batch_size)
    dict_hyperparameters["decay"] = temp_variable.learning_rate
    dict_hyperparameters["learning_rate"] = temp_variable.learning_rate
    dict_hyperparameters["momentum"] = temp_variable.momentum
    dict_hyperparameters["bool_nesterov"] = temp_variable.bool_nesterov

    dict_json["selected_model_name"] =  temp_variable.model

    # fixed values
    dict_hyperparameters["patience"] = config["hyperparameters"]["patience"]

    
    dict_hyperparameters["channels"] = config["hyperparameters"]["channels"]
    dict_hyperparameters["cropping_position"] = config["hyperparameters"]["cropping_position"]
    dict_hyperparameters["do_cropping"] = config["hyperparameters"]["do_cropping"]
    dict_hyperparameters["epochs"] = config["hyperparameters"]["epochs"]
    
    dict_json["hyperparameters"] = dict_hyperparameters
    
    dict_json["data_input_directory"] = config["data_input_directory"]
    dict_json["output_path"] = temp_variable.output_path.as_posix()
    dict_json["job_name"] = temp_variable.job_name
    dict_json["k_epoch_checkpoint_frequency"] = config["k_epoch_checkpoint_frequency"]
    dict_json["shuffle_the_images"] = config["shuffle_the_images"]
    dict_json["shuffle_the_folds"] = config["shuffle_the_folds"]
    dict_json["seed"] = config["seed"]
    dict_json["class_names"] = config["class_names"]   
    dict_json["subject_list"] = config["subject_list"]
    dict_json["test_subjects"] = config["test_subjects"]
    dict_json["validation_subjects"] = config["validation_subjects"]
    dict_json["image_size"] = config["image_size"]
    dict_json["target_height"] = config["target_height"]
    dict_json["target_width"] = config["target_width"]

    return dict_json


def create_json_file(index, dict_json, config):
    
    file_name = f"rs_{index}_config.json"
    path_output_directory_config = Path(config["configurations_directory"])
    path_output_directory_config.mkdir(mode=0o777, parents=True,
                      exist_ok=True)

    path_output_file = path_output_directory_config / file_name
       
    with open(path_output_file, "w") as output:
        json.dump(dict_json, output, indent=4)


def get_combinations(config):
    
    np.random.seed(config["seed"])
    
    a_batch_size = get_array_batch(config)
    a_learning_rate = get_array_lr(config)
    
    a_momentum = np.array(config["hyperparameters"]["l_momentum"])
    a_nesterov = np.array(config["hyperparameters"]["l_nesterov"])
    a_models = np.array(config["hyperparameters"]["l_models"])


    variables = namedtuple("random_variables",
                           ["batch_size", "learning_rate", "decay",
                            "momentum", "bool_nesterov", "model",
                            "output_path", "job_name"])


    df_summary = pd.DataFrame(columns = ["index", "model","batch_size",
                                         "learning_rate", "decay",
                                         "momentum", "bool_nesterov"])

    for index in range(config["n_trials"]):
        # random
        batch_size = np.random.choice(a_batch_size)
        learning_rate = np.random.choice(a_learning_rate)
        decay = learning_rate
        
        momentum = np.random.choice(a_momentum)
        bool_nesterov = bool(np.random.choice(a_nesterov))
        model = np.random.choice(a_models)
        
        path_output_directory = Path(config["output_path"])
        path_output_subdirectory = path_output_directory / f"random-search_{index}"
      
        output_path = path_output_subdirectory
        job_name = "random-search_{index}"
        
        df_temp = pd.DataFrame({"index":[index],
                                "model": [model],
                                "batch_size": [int(batch_size)],
                                "learning_rate": [learning_rate],
                                "decay": [decay],
                                "momentum": [momentum],
                                "bool_nesterov": [bool_nesterov] })
        
        df_summary = pd.concat([df_summary, df_temp], ignore_index=True)
        
        temp_variable = variables(batch_size, learning_rate, decay,
                                  momentum, bool_nesterov, model,
                                  output_path, job_name)        

        # fixed
        dict_json_temp = create_dict_json(temp_variable, config)
        create_json_file(index, dict_json_temp, config)

    filename_summary = "random_search_summary.csv"
    path_output_directory_config = Path(config["configurations_directory"])
    path_output_file_summary = path_output_directory_config / filename_summary  
    
    df_summary.to_csv(path_output_file_summary)
               

def main():
    config = parse_json("")
    get_combinations(config)

    
if __name__ == "__main__":
    main()