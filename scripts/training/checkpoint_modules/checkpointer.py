from tensorflow.keras.callbacks import ModelCheckpoint
from termcolor import colored
from keras import models
import keras
import os


class Checkpointer(keras.callbacks.Callback):
    def __init__(self, k_epochs, model_name="model", save_path="./"):
        """ This will create a custom checkpoint for every k epochs.
        TODO: Unused. Remove?

        Args:
            k_epochs (int): The interval of epochs at which to save.
            model_name (str, optional): The checkpoint file prefix. Defaults to "model".
            save_path (str, optional): Where the checkpoint is saved to. Defaults to "./".
        """ 
        super().__init__()
        self.k_epochs = k_epochs
        self.model_name = model_name
        self.save_path = save_path
        self.previous_save = None
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.k_epochs == 0:
            if self.previous_save:
                os.remove(self.previous_save)
            current_save = os.path.join(self.save_path, f"{self.model_name}_{epoch}.hd5")
            self.model.save(current_save)
            self.previous_save = current_save


def create_checkpoint(checkpoint_path, file_name, model_name, k_epochs):
    """ Writes a checkpoint of the current model state.

    Args:
        checkpoint_path (str): The path to the checkpoint-output location.
        file_name (str): The name of the file to produce.

    Returns:
        keras.callbacks.ModelCheckpoint: A checkpoint to save to.
    """
    # Create result folders if they don't exist
    folder_split = [f for f in checkpoint_path.split('/')]
    for folder_index in range(len(folder_split)):
        path_part = folder_split[0]
        for i in range(1, folder_index): path_part += f'/{folder_split[i]}'
        if not os.path.exists(path_part): os.makedirs(path_part)
        
    # Create checkpoint
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, f"{model_name}_{file_name}" + "_{epoch}.h5"),
        save_freq=k_epochs
    )
    return checkpoint


def load_model(path):
    """ Loads a module from a Keras checkpoint.

    Args:
        path (str): The path to the checkpoint.

    Raises:
        Exception: When no checkpoint exists.

    Returns:
        keras.Model: A loaded model.
    """
    if not os.path.exists(path):
        raise Exception(colored(f"Error: No checkpoint was found at '{path}'", 'red'))
    return models.load_model(path)
