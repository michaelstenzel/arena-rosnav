import numpy as np
import os

import torch as th
from torch.utils.data.dataset import Dataset

#TODO docstring
# get the observations and actions from the recordings
# each folder contains files with recordings generated in that map
# each file corresponds to one episode/path
# each file contains a two arrays: observations and actions
# the index of each array is the timestep in that episode

class EpisodeDataset(Dataset):
    #TODO docstring
    """ EpisodeDataset
    """
    def __init__(self, path_to_file):
        data = np.load(path_to_file)
        self.observations = data['observations']
        self.actions = data['actions']

    def __getitem__(self, timestep):
        return self.observations[timestep], self.actions[timestep]

    def __len__(self):
        return len(self.actions)

class MapDataset(Dataset):
    #TODO docstring
    """ MapDataset
    A MapDataset provides access to all episodes recorded in a given map
    self.map_dataset is a list of EpisodeDataset objects
    """
    def __init__(self, absolute_path_to_folder):
        # grab all files in path_to_folder  #TODO this grabs ALL files - change this to only grab npz files
        self.map_dataset = []
        for root, _, files in os.walk(absolute_path_to_folder):
            for file in files:
                self.map_dataset.append(EpisodeDataset(os.path.join(root, file)))

    def __getitem__(self, index):
        # self.episodes[index]
        return self.map_dataset[index]

    def __len__(self):
        # return len(self.episodes)
        return len(self.map_dataset)