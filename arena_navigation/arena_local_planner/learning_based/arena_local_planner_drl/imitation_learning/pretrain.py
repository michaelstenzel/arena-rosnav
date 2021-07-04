import os
import rospy
import rospkg
from datetime import datetime
import time

import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import random_split

from stable_baselines3 import PPO

from rl_agent.envs.flatland_gym_env import FlatlandEnv
from dataset import MapDataset

from tensorboardX import SummaryWriter


def pretrain(agent, map_dataset, num_epochs=1, batch_size=32, gamma=0.7, learning_rate=1.0):
    # the map data_set is a list of EpisodeDatasets
    # each EpisodeDatasets is a list of tuples: (observations, actions)

    #TODO add random seed for reproducibility?
    #TODO add early stopping to reduce risk of overfitting - if tensorboard logs show test loss increasing while training loss decreases
    #TODO experiment with the following hyperparameters:
    # 1. batch_size (how small should the batch size be given that the training sets are 50-200 samples?)
    # 2. gamma (learning rate scheduler)
    # 3. step_size (learning rate scheduler)
    # 4. learning rate (optimizer)
    date_str = datetime.now().strftime('%Y%m%d_%H-%M')
    writer = SummaryWriter(f'tensorboard_logs/{date_str}')

    network = agent.policy # copy network from agent
    print(f"network: {network}")

    loss_fn = nn.MSELoss() # use mean-squared error loss for regression
    # initialize optimizer
    optimizer = optim.Adadelta(network.parameters(), lr=learning_rate)  #TODO experiment with Adadelta and Adam
    # initialize learning rate scheduler
    #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)  #TODO experiment with optim.lr_scheduler.ReduceLROnPlateau

    # do 70%/30% train/test split with a random seed for reproducibility
    training_set_size = int(0.7*len(map_dataset))
    test_set_size = len(map_dataset) - training_set_size
    training_set, test_set = random_split(map_dataset, [training_set_size, test_set_size], generator=th.Generator().manual_seed(0))
    
    # batch counters: count+1 for each batch loaded from the training or test set
    # used only to log the batch loss
    training_batch_counter = 0
    test_batch_counter = 0
    for epoch in range(num_epochs):
        # TRAINING LOOP
        # iterate over the episodes of the training set:
        epoch_training_loss = 0  # keep running total of loss over the current epoch
        network.train()
        for episode in training_set:
            # instantiate dataloader for this episode
            episode_loader = th.utils.data.DataLoader(dataset=episode, batch_size=batch_size, shuffle=True)
            # iterate over timesteps in this episode
            for batch, (observation, action) in enumerate(episode_loader):
                optimizer.zero_grad()
                action_network, _, _ = network(observation)  # run forward pass to get network's prediction  #TODO check number of outputs
                #print(f"action_network: {action_network}")
                action_network = action_network.double()  # MSELoss() expects inputs to be doubles
                #print(f"action_network.double(): {action_network}")
                loss = loss_fn(action_network, action)  # compute loss

                epoch_training_loss += loss.item()
                # log average batch loss: total loss in this batch normalized by the number of timesteps in the batch
                writer.add_scalar('training batch loss', loss.item()/len(observation), training_batch_counter)
                training_batch_counter += 1

                loss.backward()  # backpropagate loss
                optimizer.step()  # adjust network parameters using the losses  #TODO check this
                
        # average training loss for the epoch: running total of loss over the epoch divided by number of samples in training_set
        writer.add_scalar('training epoch loss', epoch_training_loss/len(training_set), epoch)
        print(f"epoch {epoch}/{num_epochs} | training epoch loss: {epoch_training_loss/len(training_set)}")

        # TEST LOOP
        # iterate over the episodes of the test set to get the average test set loss
        epoch_test_loss = 0
        network.eval()  #TODO don't update gradients? Need both eval and no_grad?
        with th.no_grad():  # don't update gradients!
            for episode in test_set:
                # instantiate dataloader for this episode
                episode_loader = th.utils.data.DataLoader(dataset=episode, batch_size=batch_size, shuffle=True)
                # iterate over timesteps in this episode
                for batch, (observation, action) in enumerate(episode_loader):
                    action_network, _, _ = network(observation)
                    action_network = action_network.double()
                    loss = loss_fn(action_network, action)  #TODO check me
                    
                    epoch_test_loss += loss.item()
                    # log average batch loss: total loss in this batch normalized by the number of timesteps in the batch
                    writer.add_scalar('test batch loss', loss.item()/len(observation), test_batch_counter)
                    test_batch_counter += 1
                    
            # average test loss for the epoch: running total of loss over the epoch divided by number of samples in test_set
            writer.add_scalar('test epoch loss', epoch_test_loss/len(test_set), epoch)
            print(f"epoch {epoch}/{num_epochs} | test epoch loss: {epoch_test_loss/len(test_set)}")
        #scheduler.step()  # reduce the learning rate at the end of the episode
    
    agent.policy = network  # overwrite agent's network with the new pretrained one    
    writer.close()
    return agent


if __name__ == '__main__':  
    # need dummy scenario to instantiate FlatlandEnv. Need a gym env to instantiate a PPO agent.
    scenario = '/home/michael/catkin_ws/src/arena-rosnav/gui/eval_scenarios/0_done_json_files/eval2_with_map_empty_small/07.0__2_static_2_dynamic_obs_robot_pos1.json'
    models_folder_path = rospkg.RosPack().get_path('simulator_setup')
    arena_local_planner_drl_folder_path = rospkg.RosPack().get_path('arena_local_planner_drl')

    ns = ''

    env = FlatlandEnv(ns=ns, PATHS={'robot_setting': os.path.join(models_folder_path, 'robot', 'myrobot.model.yaml'), 'robot_as': os.path.join(arena_local_planner_drl_folder_path,
                               'configs', 'default_settings.yaml'), "model": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/rule_00",
                               "scenerios_json_path": scenario,
                               "curriculum": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/configs/training_curriculum.yaml"},
                               reward_fnc="rule_00", is_action_space_discrete=False, debug=False, train_mode=True, max_steps_per_episode=100,
                               safe_dist=None, curr_stage=1,
                               move_base_simple=True
                  )

    # create map dataset
    #map_dataset = MapDataset('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/output/map_empty_small')
    map_dataset = MapDataset('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/output/')

    # create PPO agent
    ppo_agent = PPO('MlpPolicy', env, verbose=1)  #TODO verbose=1?
    date_str = datetime.now().strftime('%Y%m%d_%H-%M')
    ppo_agent.save(f'baseline_ppo_agent_{date_str}')  # save untrained agent to use as a baseline

    # pretrain the PPO agent
    trained_agent = pretrain(ppo_agent, map_dataset, num_epochs=40, batch_size=32, learning_rate=1.0)
    print("trained agent!")

    # save the pretrained PPO agent
    date_str = datetime.now().strftime('%Y%m%d_%H-%M')
    trained_agent.save(f'pretrained_ppo_agent_{date_str}')
