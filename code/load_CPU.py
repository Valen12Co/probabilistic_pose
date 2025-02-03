# System imports
import os
import copy
import logging
from tqdm import tqdm
from typing import Union

from matplotlib import pyplot as plt#for visualize
import cv2
import random

# Science-y imports
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# File imports
from config import ParseConfig
from dataloader_CPU import load_hp_dataset, HumanPoseDataLoader


# Global declarations
logging.getLogger().setLevel(logging.INFO)
os.chdir(os.path.dirname(os.path.realpath(__file__)))

training_methods = ['MSE', 'Diagonal', 'NLL', 'Beta-NLL', 'Faithful', 'TIC']

def main() -> None:
    """
    Control flow for the code
    """

    # 1. Load configuration file ----------------------------------------------------------------------------
    logging.info('Loading configurations.\n')

    conf  = ParseConfig()


    num_hm = conf.architecture['aux_net']['num_hm']
    epochs = conf.experiment_settings['epochs']
    trials = conf.trials

    training_pkg = dict()
    for method in training_methods:
        training_pkg[method] = dict()
        training_pkg[method]['tac'] = torch.zeros((trials, num_hm), dtype=torch.float32)
        training_pkg[method]['ll'] = torch.zeros(trials, dtype=torch.float32)
        training_pkg[method]['loss'] = torch.zeros((trials, epochs))
    training_pkg['training_methods'] = training_methods 
    
    # 2. Loading datasets -----------------------------------------------------------------------------------
    logging.info('Loading pose dataset(s)\n')
    dataset_dict = load_hp_dataset(dataset_conf=conf.dataset, load_images=conf.load_images)

    # 3. Defining DataLoader --------------------------------------------------------------------------------
    logging.info('Defining DataLoader.\n')
    dataset = HumanPoseDataLoader(dataset_dict=dataset_dict, conf=conf)

main()