from sklearn.metrics import confusion_matrix
from util.get_config import parse_json
from termcolor import colored
import pandas as pd
import os




# TODO: include preprocessing of the prob values (can be found in E3b-Processing predictions first few cells)


def get_data(pred_path, true_path):
    """ Reads in the labels and predictions from CSV.

    Args:
        pred_path (str): The path to an indexed prediction file.
        true_path (str): The patth to an indexed truth file.

    Returns:
        pandas.Dataframe: Two pandas dataframes of prediction and true values.
    """
    # Read CSV file
    pred = pd.read_csv(pred_path, header=None).to_numpy()
    true = pd.read_csv(true_path, header=None).to_numpy()

    # Get shapes
    pred_rows = pred.shape[0]
    true_rows = true.shape[0]

    # Make the number of rows equal, in case uneven
    if pred_rows > true_rows:
        print(colored("Warning: The number of predicted values is greater than the true values in " +
                      f"{pred_path.split('/')[-3]}: \n\tTrue: {true_rows} | Predicted: {pred_rows}",
                      'yellow'))
        pred = pred[:true_rows, :]
    elif pred_rows < true_rows:
        print(colored("Warning: The number of true values is greater than the predicted values in " +
                      f"{pred_path.split('/')[-3]}: \n\tTrue: {true_rows} | Predicted: {pred_rows}",
                      'yellow'))
        true = true[:pred_rows, :]

    # Return true and predicted values
    return true, pred


def create_confusion_matrix(true_vals, pred_vals, results_path, file_name, labels):
    """ Creates confusion matrix and saves as a csv in the results directory.

    Args:
        true_vals (pandas.Dataframe): An array of true values.
        pred_vals (pandas.Dataframe): An array of predicted values.
        results_path (str): The path to write a matrix to.
        file_name (str): The name of the matrix file.
        labels (list(str)): A list of the labels associated with the classification index values.

    Raises:
        Exception: When true and predicted values are not equal in length.

    Returns:
        pandas.Dataframe: The created confusion matrix.
    """
    # Check the input is valid
    if len(true_vals) != len(pred_vals):
        raise Exception(colored(f'The length of true and predicted values are not equal: \n' +
                                f'\tTrue: {len(true_vals)} | Predicted: {len(pred_vals)}', 'red'))

    # Create the matrix
    conf_matrix = confusion_matrix(true_vals, pred_vals)
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=labels, index=labels)

    # Create the extra-index/col names
    conf_matrix_df.index = [["Truth"] * len(labels), conf_matrix_df.index]
    conf_matrix_df.columns = [["Predicted"] * len(labels), conf_matrix_df.columns]

    # Output the results
    conf_matrix_df.to_csv(f"{os.path.splitext(os.path.join(results_path, file_name))[0]}_{pred_vals.shape[0]}_conf_matrix.csv")

    print(colored("Confusion matrix created for " + file_name, 'green'))
    return conf_matrix


def main(config=None):
    """ The main program.

    Args:
        config (dict): A JSON configuration as a dictionary. (Optional)
    """
    # Check that the output path exists
    if config is None:
        config = parse_json(os.path.abspath('confusion_matrix_config.json'))
    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Obtain needed labels and predictions
    pred_path = config['pred_path']
    true_path = config['true_path']
    if not os.path.exists(pred_path):
        raise Exception(colored("Error: The prediction path is not valid!: " + pred_path, 'red'))
    if not os.path.exists(true_path):
        raise Exception(colored("Error: The true-value path is not valid!: " + true_path, 'red'))
    true_val, pred_val = get_data(pred_path, true_path)
    create_confusion_matrix(true_val, pred_val, output_path, config['output_file_prefix'], config['label_types'])


if __name__ == "__main__":
    """ Executes the program """
    main()
