from termcolor import colored
import dill
import os


def read_log_items(training_output_path, job_name, item_list):
    """ Reads in a log file.

    Args:
        training_output_path (str): The output path of the training results.
        job_name (str): The prefix of the log.
        item_list (list of str): A list of dictionary keys.

    Returns:
        dict: Of the specified items.
    """
    log = read_log(training_output_path, job_name)
    if not log:
        return None
    specified_items = {}
    for key in item_list:
        if key in log:
            specified_items[key] = log[key]
    return specified_items


def read_log(training_output_path, job_name):
    """ Reads in a log file.

    Args:
        training_output_path (str): The output path of the training results.
        job_name (str): The prefix of the log.

    Returns:
        tuple: Of the unpickled log. If no log, it will return None.
    """
    log_path = os.path.join(training_output_path, 'logging', f'{job_name}.log')
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return None
    else:
        try:
            with open(log_path, 'rb') as fp:
                return dill.load(fp, encoding='latin1')
        except:
            print(colored(f"Warning: Unable to open '{log_path}'", 'yellow'))
            return None
        
        
def _writing_prep(training_output_path, job_name):
    """ Gets the logging path and current log.

    Args:
        training_output_path (str): The output path of the training results.
        job_name (str): The prefix of the log.
    """
    # Check if the output directory exists
    logging_path = os.path.join(training_output_path, 'logging')
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
        
    # Check if the log already exists
    current_log = read_log(training_output_path, job_name)
    log_path = os.path.join(logging_path, f'{job_name}.log')
    return log_path, current_log
        
        
def add_to_log_item_list(training_output_path, job_name, data_dict):
    """ Writes a log of the state to file, adding to some list of values.

    Args:
        training_output_path (str): The output path of the training results.
        job_name (str): The prefix of the log.
        data_dict (dict): A dictionary of the relevant status info.
    """
    # Get the log path and value
    log_path, current_log = _writing_prep(training_output_path, job_name)
    
    # Write info
    with open(log_path, 'wb') as fp:
        
        # Write each element given in the dict to the output dictionary
        if current_log == None:
            current_log = {}
        for key in data_dict:
            if key not in current_log:
                current_log[key] = data_dict[key]
            else:
                current_log[key].extend(data_dict[key])
        dill.dump(current_log, fp, encoding='latin1')
        
        
def write_log(training_output_path, job_name, data_dict):
    """ Writes a log of the state to file. Can add individual items.

    Args:
        training_output_path (str): The output path of the training results.
        job_name (str): The prefix of the log.
        data_dict (dict): A dictionary of the relevant status info.
    """
    # Get the log path and value
    log_path, current_log = _writing_prep(training_output_path, job_name)
    
    # Write info
    with open(log_path, 'wb') as fp:
        
        # Write each element given in the dict to the output dictionary
        if current_log == None:
            current_log = {}
        for key in data_dict:
            current_log[key] = data_dict[key]
        dill.dump(current_log, fp)
        
        
def delete_log(training_output_path, job_name):
    """ Deletes a log file.

    Args:
        training_output_path (str): The output path of the training results.
        job_name (str): The prefix of the log.
    """
    # Check if the log file exists, remove if it does
    log_path = os.path.join(training_output_path, 'logging', f'{job_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)    
