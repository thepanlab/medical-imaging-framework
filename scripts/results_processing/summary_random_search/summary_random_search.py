import os 
import pathlib
import math
import json
import pandas as pd
from termcolor import colored
from scipy.stats import sem as std_err
from sklearn.metrics import accuracy_score
from util import path_getter
from util.get_config import parse_json
from results_processing.metrics_table import metrics_table
from results_processing.epoch_counting import epoch_counting


def convert_metrics_table(output_path, output_filename):
    """ 
    It finds `output_filename` under the `output_path` directory.
    Then it add `test` to the indices and  `val` to the column names.
    
    Additionally, it discard NaN values
    
    It returns modified pandas dataframe

    Args:
        output_path (string): metrics_table complete path.
        output_filename (bool): filename of the metric_table file.

    Returns:
        pandas dataframe:
    """   
    
    l_metrics_table_paths = list(output_path.glob(f"**/{output_filename}*.csv"))
    
    if len(l_metrics_table_paths) != 1:
        print(colored("More than one path for metrics table:", "red"))
        for path_temp in l_metrics_table_paths:
            print(colored(path_temp, "red"))
            
    path_metrics_csv = l_metrics_table_paths[0]
    
    df_metrics_table = pd.read_csv(path_metrics_csv, index_col=0)
    df_metrics_table = df_metrics_table.iloc[:-1, :-2]
    
    dict_col = {}
    for col in df_metrics_table.columns:
        dict_col[col] = f"val_{col}"
    
    dict_indices = {}
    for index in df_metrics_table.index.values:
        dict_indices[index] = f"test_{index}"
    
    df_metrics_table.rename(index = dict_indices,
                            columns=dict_col,
                            inplace =True)
    
    df_output = pd.melt(df_metrics_table, ignore_index = False).reset_index().sort_values(by=["index","variable"]).reset_index(drop=True)
    
    # dropna
    df_output =df_output.dropna().reset_index(drop=True)
    
    # Change columns names
    df_output = df_output.rename(columns={"index":"test_fold",
                                          "variable":"val_fold"})
    
    # Return
    return df_output
     
def read_epoch_dataframe(output_path):
    """ 
    It finds `epoch_inner_avg_stderr.csv` under the `output_path` directory.
    Then it changes column names: avg_epochs to epochs_mean, and std_err to epochs_stderr
    It returns modified pandas dataframe
    
    Args:
        output_path (string): epoch_counting complete path.
    
    Returns:
        pandas dataframe
    """ 
    path_mean_stderr = pathlib.Path(output_path) / "epoch_inner_avg_stderr.csv"

    df_epoch_mean = pd.read_csv(path_mean_stderr)

    l_columns = ["test_fold", "avg_epochs", "std_err"]

    df_epoch_mean_subset = df_epoch_mean[l_columns]
    df_epoch_mean_subset = df_epoch_mean_subset.rename(columns={"avg_epochs": "epochs_mean",
                                                                "std_err": "epochs_stderr"})
    
    return df_epoch_mean_subset

def create_json_file(test_fold, dict_json, config):
    """ 
    It creates configuration json files with the best set of hyperparamters
    from random search and average value of epochs.
        
    Args:
        test_fold (string): test_fold name
        dict_json (dict): json in dictionary format 
        config (dict): main configuration.
    
    Returns:
        path location of json file
    """ 
    file_name = f"outer_{test_fold}_config.json"
    path_output_directory_config = pathlib.Path(config["path_output_configurations"])
    path_output_directory_config.mkdir(mode=0o777, parents=True,
                                        exist_ok=True)

    path_output_file = path_output_directory_config / file_name
       
    with open(path_output_file, "w") as output:
        json.dump(dict_json, output, indent=4)

    return path_output_file

def generate_and_combine_metrics_table(list_directories, config):
    """ 
    It runs `metrics_table` for the directories in list_directories.
    Additionally, it joins in a master table and stores in 
    directory config["output_path"] and uses 
    config["prefix_output_filename"] for naming
    
    saved cvs: (df_summary_randomsearch)
    
    | |index| test_fold|rs|  val_fold|value|
    |0|  720|test_fold1| 0|val_fold10|  1.0|
    |1|  721|test_fold1| 0| val_fold2|  1.0|
    |2|  722|test_fold1| 0| val_fold3|  1.0|
    ...
    
    Args:
        list_directories (list of strings): list with directories of random search
        config (dict): main configuration.
    
    Returns:
        pandas dataframe: 
    """ 
    df_summary_randomsearch = pd.DataFrame()
    
    for directory in list_directories:
        config_temp = {}

        data_path_directory = directory / "training_results"
        config_temp["data_path"] = str(data_path_directory)
        
        output_path = directory / "metrics_table"
        config_temp["output_path"] = output_path
        
        random_search_index = directory.name.split("_")[-1]
        output_filename = config["prefix_output_filename"] + f"_rs{random_search_index}"
        config_temp["output_filename"] = output_filename

        config_temp["round_to"] = config["round_to"]
        config_temp["is_outer"] = False

        print(colored(f"Creating metrics table for {directory} in {output_path}.", 'green'))

        metrics_table.main(config_temp)
        
        # read csv metrics_table
        
        df_temp = convert_metrics_table(output_path, output_filename)
        
        # add random search value
        
        df_temp.insert(1,"rs",random_search_index)
        
        df_summary_randomsearch = pd.concat([df_summary_randomsearch, df_temp], ignore_index=True)
        
    df_summary_randomsearch = df_summary_randomsearch.sort_values(by=["test_fold","rs","val_fold"])
    df_summary_randomsearch  = df_summary_randomsearch.reset_index()
    
    path_folder_output = pathlib.Path(config["output_path"])
    path_folder_output.mkdir(parents=True, exist_ok=True)
    
    file_path = path_folder_output / f'{config["prefix_output_filename"]}_all.csv'
    
    df_summary_randomsearch.to_csv(file_path)
    
    print(colored(f"Saving table all combinations in {file_path}.", 'green'))
    
    return df_summary_randomsearch

def generate_mean_stderr(df_summary_randomsearch, config):
    """ 
    It calculates the count, mean and stderr. It outputs two CVSs
    in different formats 
    
    
    It saves a csv file with the following columns:
    test_fold, rs, count, mean, stderr
        
    saved csv: (df_mean_stderr csv):
    
    | test_fold|rs|count|mean|stderr|
    |----------|--|-----|----|------|
    |test_fold1| 0|    9|0.95|  0.05|
    |test_fold2| 1|    9|0.85|  0.12|
    ...
   
    saved csv: (df_mean_stderr_v2):
    
    |                  | rs| rs| rs| rs| rs| ...| rs| rs|
    |                  |  0|  1|  2|  3|  4| ...|  7|  8|
    |------------------|---|---|---|---|---| ...|---|---| 
    |   test_fold1_mean|1.0|1.0|1.0|1.0|1.0| ...|0.8|0.7|
    | test_fold1_stderr|0.0|0.0|0.0|0.0|0.0| ...|0.1|0.1|
    |  test_fold10_mean|1.0|1.0|1.0|1.0|1.0| ...|0.7|0.9|
    |test_fold10_stderr|0.0|0.0|0.0|0.0|0.0| ...|0.1|0.2|
    ....
        
    Args:
        df_summary_randomsearch (pandas dataframe): dataframe with columns index,
                                                test_fold, rs, val_fold, value
        config (dict): main configuration.
        
    Returns:
        pandas dataframe: (df_mean_stderr)
    
    """ 
    # group by "test_fold" and "rs"
    # Calculate mean
    df_summary_randomsearch_mean = df_summary_randomsearch.groupby(by=["test_fold","rs"]).mean()
    df_summary_randomsearch_mean = df_summary_randomsearch_mean.drop(columns=["index"])
    df_summary_randomsearch_mean  = df_summary_randomsearch_mean.rename(columns={"value":"mean"})
    
    # std_err = std deviation / sqrt(count)
    df_summary_randomsearch_stddev = df_summary_randomsearch.groupby(by=["test_fold","rs"]).std()[["value"]]

    # Count number of values (validation)    
    df_summary_randomsearch_count = df_summary_randomsearch.groupby(by=["test_fold","rs"]).count()[["value"]]
    # to be able to divide, both dataframes column must have same name (i.e 'value')
    df_summary_randomsearch_stderror = df_summary_randomsearch_stddev.div(df_summary_randomsearch_count**0.5)
    df_summary_randomsearch_stderror = df_summary_randomsearch_stderror.rename(columns={"value":"stderr"})


    df_summary_randomsearch_count = df_summary_randomsearch_count.rename(columns={"value":"count"})

    # join in mean and std err
    df_count_mean_stderr_csv = pd.merge(df_summary_randomsearch_count, df_summary_randomsearch_mean,
                                  left_index=True, right_index=True)
    df_count_mean_stderr_csv = pd.merge(df_count_mean_stderr_csv, df_summary_randomsearch_stderror,
                                  left_index=True, right_index=True)

    path_folder_output = pathlib.Path(config["output_path"])
    path_folder_output.mkdir(parents=True, exist_ok=True)

    file_path = path_folder_output / f'{config["prefix_output_filename"]}_count_mean_stderr.csv'
        
    df_count_mean_stderr_csv.to_csv(file_path)
    
    print(colored(f"Saving table means and standard error in {file_path}.", 'green'))

    df_mean_stderr = pd.merge(df_summary_randomsearch_mean, df_summary_randomsearch_stderror,
                              left_index=True, right_index=True)

    # Creating new version of table    
    # Because of group by 
    # Indices is composed of two parts
    # "test_fold" and "rs"
    
    l_test_folds = []
    for test_fold, rs in df_mean_stderr.index.values:
        if test_fold not in l_test_folds:
            l_test_folds.append(test_fold)
        
    idx = pd.IndexSlice
    
    df_mean_stderr_v2 =pd.DataFrame()

    for test_fold in l_test_folds:
        # Transpose to have rs values as columns
        df_mean_stderr_v2_temp = df_mean_stderr.loc[idx[test_fold,:]].T
        
        df_mean_stderr_v2_temp = df_mean_stderr_v2_temp.rename(index={"mean":f"{test_fold}_mean",
                                                                      "stderr":f"{test_fold}_stderr"})
    
        df_mean_stderr_v2 = pd.concat((df_mean_stderr_v2, df_mean_stderr_v2_temp))
    
    l_tuples = [("rs", col_name) for col_name in df_mean_stderr_v2.columns]
    
    df_mean_stderr_v2.columns = pd.MultiIndex.from_tuples(l_tuples)
    
    file_path = path_folder_output / f'{config["prefix_output_filename"]}_mean_stderr_v2.csv'
        
    df_mean_stderr_v2.to_csv(file_path)
    
    print(colored(f"Saving table means and standard error v2 in {file_path}.", 'green'))

    return df_mean_stderr


def generate_epoch_csv(list_directories, config):
    """ 
    It runs `epoch_counting` for the directories in list_directories.
    Additionally, it stores in 
    directory config["output_path"] and uses 
    config["prefix_output_filename"] for naming

    It saves a csv file with the following columns:
    test_fold, rs, epochs_mean, epochs_stderr
    
    saved csv: (df_epochs_mean_stderr):
       
    | | test_fold|rs|epochs_mean|epochs_stderr|
    |-|----------|--|-----------|-------------|
    |0|test_fold1| 0|       50.0|          0.0|
    |1|test_fold1| 1|49.88888888|0.11111111111|
    |2|test_fold1| 2|       50.0|          0.0|
    |3|test_fold1| 3|       50.0|          0.0|
    |4|test_fold1| 4|       50.0|          0.0|
    ...
   
    Args:
        list_directories (list of strings): list with directories of random search
        config (dict): main configuration.
            
    Returns:
        pandas dataframe: (df_epochs_mean_stderr)
    """ 
    
    df_epochs_mean_stderr = pd.DataFrame()
    
    for directory in list_directories:
        config_temp = {}

        data_path_directory = directory / "training_results"
        config_temp["data_path"] = str(data_path_directory)
        
        output_path = directory / "epoch_counting"
        config_temp["output_path"] = output_path
        
        random_search_index = directory.name.split("_")[-1]

        config_temp["is_outer"] = False

        print(colored(f"Creating metrics table for {directory} in {output_path}.",
                      'green'))

        epoch_counting.main(config_temp)
           
        df_temp = read_epoch_dataframe(output_path)        
        # add random search value
        
        df_temp.insert(1,"rs",random_search_index)
        
        df_epochs_mean_stderr = pd.concat([df_epochs_mean_stderr, df_temp], ignore_index=True)
    
    df_epochs_mean_stderr = df_epochs_mean_stderr.sort_values(by=['test_fold',"rs"]).reset_index(drop=True)
    
    path_folder_output = pathlib.Path(config["output_path"])
    path_folder_output.mkdir(parents=True, exist_ok=True)
    
    file_path = path_folder_output / f'{config["prefix_output_filename"]}_epochs_mean_stderr.csv'
    
    l_test_folds = []
    for test_fold in df_epochs_mean_stderr["test_fold"]:
        if test_fold not in l_test_folds:
            l_test_folds.append(test_fold)
    
    # Changing name of fold* to test_fold*
    dict_change = {}

    for test_fold in  l_test_folds:
        dict_change[test_fold] = f"test_{test_fold}"
    
    df_epochs_mean_stderr = df_epochs_mean_stderr.replace(dict_change)
    
    df_epochs_mean_stderr.to_csv(file_path)
    
    print(colored(f"Saving epoch means and standard error in {file_path}.", 'green'))
    
    return df_epochs_mean_stderr


def generate_best(df_mean_stderr, df_epochs_mean_stderr,
                  config):
    """ 
    First, it generates a dataframe that combines `df_mean_stderr` and
    `df_epochs_mean_stderr`: df_merged
    
    Second, it generates a data frame with the best index for random search
    according to average performance. It several values have the same
    performance, first value according to index is chosen (be cautious!)
    
    df_merged columns:
    test_fold, rs, mean, stderr, epochs_mean, epochs_stderr
        
    saved csv: (df_merged)
    
    | test_fold|rs|mean|stderr|epochs_mean|epochs_stderr|
    |----------|--|----|------|-----------|-------------|
    |test_fold1| 0| 1.0|   0.0|       50.0|          0.0|
    |test_fold1| 1| 1.0|   0.0|49.88888886|0.11111111111|
    |test_fold1| 2| 1.0|   0.0|       50.0|          0.0|
    ...

    df_best_w_epoch:
    test_fold, rs, mean, stderr, epochs_mean, epochs_stderr, epochs_ceiling
    
    saved csv: (df_best_w_epoch)
   
    | |  test_fold|rs|epochs_mean|epochs_stderr|epochs_ceiling|
    |-|-----------|--|-----------|-------------|--------------|
    |0| test_fold1| 0|       50.0|          0.0|            50|
    |1|test_fold10| 0|       50.0|          0.0|            50|
    |2| test_fold2| 0|       50.0|          0.0|            50|
    ...
   
    Args:
        df_mean_stderr (pandas dataframe): dataframe with mean and std err of accuracy
        df_epochs_mean_stderr (pandas dataframe): dataframe with mean and std err of epoch
        config (dict): main configuration.
    
    """     
    
    df_epochs_mean_stderr_indexed = df_epochs_mean_stderr.set_index(['test_fold', 'rs'])
    
    df_merged = pd.merge(df_mean_stderr, df_epochs_mean_stderr_indexed, left_index=True, right_index=True)    

    path_folder_output = pathlib.Path(config["output_path"])
    path_folder_output.mkdir(parents=True, exist_ok=True)

    file_path = path_folder_output / f'{config["prefix_output_filename"]}_merged.csv'

    df_merged.to_csv(file_path)

    print(colored(f"Saving merged table in {file_path}.", 'green'))

    l_test_folds = []
    for test_fold, rs in df_mean_stderr.index.values:
        if test_fold not in l_test_folds:
            l_test_folds.append(test_fold)

    df_best =pd.DataFrame()
    
    for test_fold in l_test_folds:
        mean_index, _ = df_mean_stderr.loc[[test_fold]].idxmax()
        rs_index = mean_index[1]
        
        mean_temp = df_mean_stderr.loc[(test_fold, rs_index), "mean"]
        stderr_temp = df_mean_stderr.loc[(test_fold, rs_index), "stderr"]
        
        
        df_temp = pd.DataFrame({"test_fold":[test_fold],
                                "rs":[rs_index],
                               "mean": [mean_temp],
                                "stderr":[stderr_temp]})
        
        df_best = pd.concat([df_best, df_temp], ignore_index=True)

    file_path = path_folder_output / f'{config["prefix_output_filename"]}_best.csv'

    df_best = df_best.set_index(['test_fold', 'rs'])
    df_best_w_epoch = pd.merge(df_best, df_epochs_mean_stderr_indexed,
                               left_index=True, right_index=True)

    # Round to create a new column
    df_best_w_epoch["epochs_ceiling"] = df_best_w_epoch["epochs_mean"].apply(math.ceil)
    
    df_best_w_epoch = df_best_w_epoch.reset_index()

    df_best_w_epoch.to_csv(file_path)
 
    print(colored(f"Saving table best per test fold in {file_path}.", 'green'))

    return df_best_w_epoch


def create_configurations(df_best_w_epoch, config):
    """ 
    It creates json configurations for outer loop
    
  
    Args:
        df_best_w_epoch (pandas dataframe): list with directories of random search
        config (dict): main configuration.
    
    """  
    # Creation of configurations files for outer loop
    
    # get rs_0_config.json
    # different configurations will some of its parameters
    path_directories_configurations = pathlib.Path(config["path_directory_rs_configurations"])
    path_rs_0 = path_directories_configurations / "rs_0_config.json"
    
    f = open(path_rs_0)
    dict_rs_0 = json.load(f)
    f.close()

    # import table of configurations   
    path_table_rs = path_directories_configurations / "random_search_summary.csv"
    df_table_rs = pd.read_csv(path_table_rs, index_col=0)
    
    path_output_results_outer_parent = pathlib.Path(config["output_path_outer"])
    
    # building dict for outer configurations
    for test_fold in df_best_w_epoch["test_fold"]:
        df_test_best = df_best_w_epoch.query(f"test_fold == '{test_fold}'")
        
        outer_fold = test_fold.split("_")[1]
        
        rs_best = df_test_best["rs"].values[0]
        rs_best_epoch = df_test_best["epochs_ceiling"].values[0]
        
        df_rs_best = df_table_rs.query(f"index == {rs_best}")     
        
        # Create outer configuration
        
        dict_json = {}
        
        dict_hyperparameters = {}

        # random values    
        dict_hyperparameters["batch_size"] = int(df_rs_best["batch_size"].values[0])
        dict_hyperparameters["decay"] = float(df_rs_best["decay"].values[0])
        dict_hyperparameters["learning_rate"] = float(df_rs_best["learning_rate"].values[0])
        dict_hyperparameters["momentum"] = float(df_rs_best["momentum"].values[0])  
        dict_hyperparameters["bool_nesterov"] = bool(df_rs_best["bool_nesterov"].values[0]) 

        dict_json["selected_model_name"] =  df_rs_best["model"].values[0]

        # fixed values
        dict_hyperparameters["channels"] = dict_rs_0["hyperparameters"]["channels"]
        dict_hyperparameters["cropping_position"] = dict_rs_0["hyperparameters"]["cropping_position"]
        dict_hyperparameters["do_cropping"] = dict_rs_0["hyperparameters"]["do_cropping"]
        dict_hyperparameters["epochs"] = int(rs_best_epoch)
        
               
        dict_json["hyperparameters"] = dict_hyperparameters
        
        dict_json["data_input_directory"] = dict_rs_0["data_input_directory"]
        
        path_output_outer = path_output_results_outer_parent / test_fold
        
        dict_json["output_path"] = path_output_outer.as_posix()
        dict_json["job_name"] = f"outer_{test_fold}"
        dict_json["k_epoch_checkpoint_frequency"] = dict_rs_0["k_epoch_checkpoint_frequency"]
        dict_json["shuffle_the_images"] = dict_rs_0["shuffle_the_images"]
        dict_json["shuffle_the_folds"] = dict_rs_0["shuffle_the_folds"]
        dict_json["seed"] = dict_rs_0["seed"]
        dict_json["class_names"] = dict_rs_0["class_names"]   
        dict_json["subject_list"] = dict_rs_0["subject_list"]
        dict_json["test_subjects"] = [outer_fold]
        dict_json["image_size"] = dict_rs_0["image_size"]
        dict_json["target_height"] = dict_rs_0["target_height"]
        dict_json["target_width"] = dict_rs_0["target_width"]

        path_output_file = create_json_file(test_fold, dict_json, config)

        print(colored(f"Saving outer configuration for {test_fold} in {path_output_file}.", 'green'))


def main(config=None):
    """ The main program.

    Args:
        config (dict): A JSON configuration as a dictionary. (Optional)
    """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('./results_processing/metrics_table/metrics_table_config.json')
    
    # Detect if "results_folder_or_list" is a list of a single file
    
    l_directories = []
    
    # parent folder
    if type(config["results_folder_or_list"]) == list:
        l_directories =  config["results_folder_or_list"]
    # list of results
    else:
        path_folder = pathlib.Path(config["results_folder_or_list"])
        # the random search folders are just one level below
        l_directories = [f for f in path_folder.iterdir() if f.is_dir()]    
    
    # Join all metrics tables in `df_summary_randomsearch`
    
    df_summary_randomsearch = generate_and_combine_metrics_table(l_directories, config)
    
    df_mean_stderr = generate_mean_stderr(df_summary_randomsearch, config)
    
    df_epochs_mean_stderr = generate_epoch_csv(l_directories, config)
    
    df_best_w_epoch = generate_best(df_mean_stderr, df_epochs_mean_stderr,
                                    config)
    
    create_configurations(df_best_w_epoch, config)

    
if __name__ == "__main__":
    """ Executes the program """
    main()