from logger import Logger
import torch
import os
from operator import itemgetter

from torch import __init__

def checkpoint_model(PATH, model, optimizer, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {'epoch': epoch,
                             'last_global_step': last_global_step,
                             'model_state_dict': model.network.module.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict()}
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)
    torch.save(checkpoint_state_dict, PATH)
    return


def load_checkpoint(model, optimizer, PATH):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = torch.load(PATH, map_location=torch.device("cpu"))
    #from train import model
    model.network.module.load_state_dict(
        checkpoint_state_dict['model_state_dict'])
    #from train import optimizer
    optimizer.load_state_dict(checkpoint_state_dict['optimizer_state_dict'])
    epoch = checkpoint_state_dict['epoch']
    last_global_step = checkpoint_state_dict['last_global_step']
    del checkpoint_state_dict
    return (epoch + 1, last_global_step)


def latest_checkpoint_file(reference_folder: str, no_cuda) -> str:
    """Extracts the name of the last checkpoint file

    :param reference_folder: (str) Path to the parent_folder
    :return: (str) Path to the most recent checkpoint tar file
    """

    # Extract sub-folders under the reference folder
    matching_sub_dirs = [d for d in os.listdir(reference_folder)]

    logger = Logger(cuda=torch.cuda.is_available() and not no_cuda)
    
    # For each of these folders, find those that correspond
    # to the proper architecture, and that contain .tar files
    candidate_files = []
    for sub_dir in matching_sub_dirs:
        for dir_path, dir_names, filenames in os.walk(os.path.join(reference_folder, sub_dir)):
            if 'saved_models' in dir_path:
                relevant_files = [f for f in filenames if f.endswith('.tar')]
                if relevant_files:
                    latest_file = max(relevant_files)  # assumes that checkpoint number is of format 000x
                    candidate_files.append((dir_path, latest_file))
    
    checkpoint_file = max(candidate_files, key=itemgetter(1))
    checkpoint_path = os.path.join(checkpoint_file[0], checkpoint_file[1])

    return checkpoint_path
