import numpy as np
import os
from torch.utils.data.dataset import Dataset


class EpisodeDataset(Dataset):
    """ EpisodeDataset
    contains all the (observation, action) pairs in one episode/rollout/trajectory from starting position to goal.
    Args:
        path_to_file (str): path (including .npz file extension) of an npz file storing two numpy arrays:
            one called "observations" and one called "actions". The index of each array indexes the timestep in that episode.
            It is assumed that the data is synchronized, that is that at each index
            the elements of both arrays form one (observation, action) pair.
            See save_episode() in human_expert.py or record_rollouts.py to see how the files are saved.
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
    """ MapDataset
    A MapDataset provides access to all episodes recorded on a given map.
    The constructor traverses all subfolders of absolute_path_to_folder (the root directory) and adds all
    files to the MapDataset. They are assumed to all be npz files.

    self.map_dataset is a list of EpisodeDataset objects.
    
    Args:
        absolute_path_to_folder (str): absolute path to the root folder of the dataset
    """
    def __init__(self, absolute_path_to_folder):
        # grab all files in path_to_folder  #TODO this grabs ALL files - change this to only grab npz files
        self.map_dataset = []
        for root, _, files in os.walk(absolute_path_to_folder):
            for file in files:
                self.map_dataset.append(EpisodeDataset(os.path.join(root, file)))

    def __getitem__(self, index):
        return self.map_dataset[index]

    def __len__(self):
        return len(self.map_dataset)