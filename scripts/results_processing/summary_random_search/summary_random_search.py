import os 
import pathlib
import pandas as pd
from termcolor import colored
from scipy.stats import sem as std_err
from sklearn.metrics import accuracy_score
from util import path_getter
from util.get_config import parse_json
from results_processing.metrics_table import metrics_table


def convert_metrics_table(output_path, output_filename):
    
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
    
    # Change volumns names
    df_output = df_output.rename(columns={"index":"test_fold",
                                          "variable":"val_fold"})
    
    # Return
    return df_output
     

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
    
    if type(config["results_folder_or_list"]) == list:
        l_directories =  config["results_folder_or_list"]
    else:
        path_folder = pathlib.Path(config["results_folder_or_list"])
        # the random search folders are just one level below
        l_directories = [f for f in path_folder.iterdir() if f.is_dir()]    
    
    # df_summary_randomsearch = pd.DataFrame(columns = ["test_fold","random_search","validation_fold","accuracy"])
    df_summary_randomsearch = pd.DataFrame()
    
    for directory in l_directories:
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
        
        # Use values from metrics table to create new selection table
        # Create super table 
    
    df_summary_randomsearch = df_summary_randomsearch.sort_values(by=["test_fold","rs","val_fold"])

    path_folder_output = pathlib.Path(config["output_path"])
    path_folder_output.mkdir(parents=True, exist_ok=True)
    
    file_path = path_folder_output / f'{config["prefix_output_filename"]}_all.csv'
    
    df_summary_randomsearch.to_csv(file_path)
    
    print(colored(f"Saving table all combinations in {file_path}.", 'green'))
        
    df_summary_randomsearch_mean = df_summary_randomsearch.groupby(by=["test_fold","rs"]).mean()
    df_summary_randomsearch_mean  = df_summary_randomsearch_mean.rename(columns={"value":"mean"})
    
    df_summary_randomsearch_stderror = df_summary_randomsearch.groupby(by=["test_fold","rs"]).std().div(df_summary_randomsearch.groupby(by=["test_fold","rs"]).count()[["value"]]**0.5)
    df_summary_randomsearch_stderror  = df_summary_randomsearch_stderror.rename(columns={"value":"stderr"})
    
    df_mean_stderr = pd.merge(df_summary_randomsearch_mean, df_summary_randomsearch_stderror, left_index=True, right_index=True)

    file_path = path_folder_output / f'{config["prefix_output_filename"]}_mean_stderr.csv'
        
    df_mean_stderr.to_csv(file_path)
    
    print(colored(f"Saving table means and standard error in {file_path}.", 'green'))
    
    l_test_folds = []
    # Select the best 
    for index in df_mean_stderr.index.values:
        
        if index[0] not in l_test_folds:
            l_test_folds.append(index[0])
    
    df_best =pd.DataFrame()
    
    for test_fold in l_test_folds:
        mean_index, _ = df_mean_stderr.loc[[test_fold]].idxmax()
        rs_index = mean_index[1]
        
        df_temp = pd.DataFrame({"test_fold":[test_fold],
                                "best_rs":[rs_index]})
        df_best = pd.concat([df_best, df_temp], ignore_index=True)

    file_path = path_folder_output / f'{config["prefix_output_filename"]}_mean_best.csv'

    df_best.to_csv(file_path)
 
    print(colored(f"Saving table best per test fold in {file_path}.", 'green'))
    
        
if __name__ == "__main__":
    """ Executes the program """
    main()