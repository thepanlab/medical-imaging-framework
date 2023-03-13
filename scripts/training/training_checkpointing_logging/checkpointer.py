from tensorflow.keras.callbacks import ModelCheckpoint
from termcolor import colored
from keras import models
import fasteners
import keras
import os


class Checkpointer(keras.callbacks.Callback):
    def __init__(self, n_epochs, k_epochs, file_name, save_path="./"):
        """ This will create a custom checkpoint for every k epochs.
        Args:
            n_epochs (int): The total expected epochs.
            k_epochs (int): The interval of epochs at which to save.
            file_name (str): The model metadata. Contains the job name, config name, and subjects.
            save_path (str, optional): Where the checkpoint is saved to. Defaults to "./".
        """ 
        super().__init__()
        self.n_epochs = n_epochs
        self.k_epochs = k_epochs
        self.file_name = file_name
        self.save_path = save_path
        self.prev_save = None
        
        
    def on_epoch_end(self, epoch, logs=None):
        # If the epoch is within k steps, save it
        if epoch % self.k_epochs == 0 or (epoch+1) == self.n_epochs:
            
            with fasteners.InterProcessLock(os.path.join(self.save_path, 'ckpt_lock.tmp')):
                new_save_path = os.path.join(self.save_path, f"{self.file_name}_{epoch+1}.h5")
                self.model.save(new_save_path)
                print(colored(f"\nSaved a checkpoint for epoch {epoch+1}/{self.n_epochs}.", 'cyan'))
            
            # Keep only the previous checkpoint for the most recent training fold, to save memory.
            self.clear_prev_save(new_save_path)
        
        
    def clear_prev_save(self, new_save_path=None):
        # If a previous save exists, delete it
        if self.prev_save:
            os.remove(self.prev_save)
        if new_save_path:
            self.prev_save = new_save_path
        else:
            self.prev_save = None
            

def get_most_recent_checkpoint(save_path, file_prefix, get_epoch=True):
    checkpoints = [os.path.join(save_path, file) for file in os.listdir(save_path) if file.startswith(file_prefix)]
    if len(checkpoints) == 0:
        return None
    elif len(checkpoints) > 1:
        print(colored("Warning: Multiple checkpoints exist with the same file name prefix. Using the one with the greatest epoch...", 'yellow'))
    
    checkpoint_epochs = [int(os.path.splitext(os.path.basename(chkpt))[0].split('_')[-1]) for chkpt in checkpoints]
    max_epoch = max(checkpoint_epochs)
    model = load_checkpoint(checkpoints[checkpoint_epochs.index(max_epoch)], False)
    if get_epoch:
        return model, max_epoch
    return model
        


def load_checkpoint(path, get_epoch=True):
    """ Loads a model from a Keras checkpoint.

    Args:
        path (str): The path to the checkpoint.
        get_epoch (str): Whether to return the epoch number from the checkpoint name. Default is True. (Optional)

    Raises:
        Exception: When no checkpoint exists.

    Returns:
        keras.Model: A loaded model.
    """
    if not os.path.exists(path):
        raise Exception(colored(f"Error: No checkpoint was found at '{path}'", 'red'))
    if get_epoch:
        epoch = path.split('/')[-1].split('.')[0].split('_')[-1]
        return models.load_model(path), epoch
    return models.load_model(path)
