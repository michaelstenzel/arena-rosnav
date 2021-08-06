import os
import argparse
import rospkg
from datetime import datetime

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import random_split

from stable_baselines3 import PPO

from rl_agent.envs.flatland_gym_env import FlatlandEnv
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.scripts.custom_policy import *
from dataset import MapDataset

from tensorboardX import SummaryWriter

"""
pretrain.py

Trains an agent using supervised learning. The agent is either an existing one, or a randomly initialized AGENT18.
Mean-squared error is used as 

Args:
            task (ABSTask): [description]
            reward_fnc (str): [description]
            train_mode (bool): bool to differ between train and eval env during training
            is_action_space_discrete (bool): [description]
            safe_dist (float, optional): [description]. Defaults to None.
            goal_radius (float, optional): [description]. Defaults to 0.1.
            extended_eval (bool): more episode info provided, no reset when crashing
"""


def pretrain(agent, map_dataset, num_epochs=5, batch_size=15, gamma=0.7, learning_rate=1.0, dataset='human_expert'):
    date_str = datetime.now().strftime('%Y%m%d_%H-%M')
    writer = SummaryWriter(f'tensorboard_logs/{dataset}/{date_str}')

    network = agent.policy # copy network from agent
    print(f"network: {network}")
    print('beginning training - epoch 0')

    loss_fn = nn.MSELoss() # use mean-squared error loss for regression
    # initialize optimizer
    optimizer = optim.Adadelta(network.parameters(), lr=learning_rate)

    # do 70%/30% train/test split with a random seed for reproducibility
    training_set_size = int(0.7*len(map_dataset))
    test_set_size = len(map_dataset) - training_set_size
    training_set, test_set = random_split(map_dataset, [training_set_size, test_set_size], generator=th.Generator().manual_seed(0))

    # initialize dataloaders to iterate over entire train and test sets. shuffle=True, so all samples are shuffled at the end of each epoch.
    training_episode_list = [episode for episode in training_set]  # create list of all EpisodeDatasets in training_set
    concat_train_set = th.utils.data.ConcatDataset(training_episode_list)  # concatenate episodes in the list: each element of concat_train_set is one (observation, action) pair
    train_loader = th.utils.data.DataLoader(dataset=concat_train_set, batch_size=batch_size, shuffle=True)  # train_loader returns mini-batches of (observation, action) pairs.
    test_episode_list = [episode for episode in test_set]
    concat_test_set = th.utils.data.ConcatDataset(test_episode_list)
    test_loader = th.utils.data.DataLoader(dataset=concat_test_set, batch_size=batch_size, shuffle=True)
    
    # batch counters: count+1 for each batch loaded from the training or test set
    # used only to log the batch loss
    training_batch_counter = 0
    test_batch_counter = 0
    for epoch in range(num_epochs):
        # TRAINING LOOP
        epoch_training_loss = 0  # keep running total of loss over the current epoch
        network.train()
        for batch, (observation, action) in enumerate(train_loader):
            optimizer.zero_grad()
            action_network, _, _ = network(observation)  # run forward pass to get network's prediction
            action_network = action_network.double()  # MSELoss() expects inputs to be doubles
            loss = loss_fn(action_network, action)  # compute loss

            epoch_training_loss += loss.item()
            # log average batch loss: total loss in this batch normalized by the number of timesteps in the batch
            writer.add_scalar('training batch loss', loss.item()/len(observation), training_batch_counter)
            training_batch_counter += 1

            loss.backward()  # backpropagate loss
            optimizer.step()  # adjust network parameters using the losses
                
        # average training loss for the epoch: running total of loss over the epoch divided by number of samples in training_set
        writer.add_scalar('training epoch loss', epoch_training_loss/len(training_set), epoch)
        print(f"epoch {epoch}/{num_epochs} | training epoch loss: {epoch_training_loss/len(training_set)}")

        # TEST LOOP
        epoch_test_loss = 0
        network.eval()
        with th.no_grad():  # don't update gradients
            for batch, (observation, action) in enumerate(test_loader):
                action_network, _, _ = network(observation)
                action_network = action_network.double()
                loss = loss_fn(action_network, action)
                
                epoch_test_loss += loss.item()
                # log average batch loss: total loss in this batch normalized by the number of timesteps in the batch
                writer.add_scalar('test batch loss', loss.item()/len(observation), test_batch_counter)
                test_batch_counter += 1
                    
            # average test loss for the epoch: running total of loss over the epoch divided by number of samples in test_set
            writer.add_scalar('test epoch loss', epoch_test_loss/len(test_set), epoch)
            print(f"epoch {epoch}/{num_epochs} | test epoch loss: {epoch_test_loss/len(test_set)}")
    
    agent.policy = network  # overwrite agent's network with the new pretrained one    
    writer.close()
    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('-e', '--num_epochs', type=int, help='number of training epochs', default=5)
    parser.add_argument('-bs', '--batch_size', type=int, help='path of scenario json file for deployment', default=15)
    parser.add_argument('-lr', '--learning_rate', type=float, help='initial learning rate', default=1.0)
    parser.add_argument('-ds', '--dataset', type=str, help='string describing the dataset (e.g. "mpc" or "human_expert"). We be used for tensorboard logs and output agent files.', default='human_expert')
    parser.add_argument('-load', '--load', type=str, help='supply FILENAME AND PATH RELATIVE TO /arena_local_planner_drl of an existing agent file for continued training. in not set, a randomly initialized AGENT18 will be used. e.g. "agents/AGENT_1_2021_04_02__22_03/best_model.zip"', default=None)
    
    args = parser.parse_args()
    
    # need dummy scenario to instantiate FlatlandEnv. Need a gym env to instantiate a PPO agent.
    scenario = '/home/michael/catkin_ws/src/arena-rosnav/gui/eval_scenarios/0_done_json_files/eval2_with_map_empty_small/07.0__2_static_2_dynamic_obs_robot_pos1.json'
    models_folder_path = rospkg.RosPack().get_path('simulator_setup')
    arena_local_planner_drl_folder_path = rospkg.RosPack().get_path('arena_local_planner_drl')

    ns = ''

    #TODO passing a "scenerios_json_path" may no longer be necessary
    env = FlatlandEnv(ns=ns, PATHS={'robot_setting': os.path.join(models_folder_path, 'robot', 'myrobot.model.yaml'), 'robot_as': os.path.join(arena_local_planner_drl_folder_path,
                               'configs', 'default_settings.yaml'), "model": os.path.join(arena_local_planner_drl_folder_path, 'agents', 'rule_04'),
                               "scenerios_json_path": scenario,
                               "curriculum": os.path.join(arena_local_planner_drl_folder_path, "configs" , "training_curriculum_map1small.yaml")},
                               reward_fnc="rule_04", is_action_space_discrete=False, debug=False, train_mode=True, max_steps_per_episode=600,
                               safe_dist=None, curr_stage=1,
                               move_base_simple=False
                  )

    # create map dataset
    #map_dataset = MapDataset('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/output/map_empty_small')
    # output folder: MPC
    #map_dataset = MapDataset('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/output/')

    # human expert folder:
    map_dataset = MapDataset(f'{arena_local_planner_drl_folder_path}/imitation_learning/output/human_expert')

    if args.load:
        # load an existing agent to continue training on map_dataset
        print(f'loading existing agent: {args.load}')
        ppo_agent = PPO.load(os.path.join(arena_local_planner_drl_folder_path, args.load), env)
    else:
        # instantiate AGENT18 (CNN), will have randomly initialized weights
        print(f'randomly initializing AGENT18')
        policy_kwargs = policy_kwargs_agent_18
        ppo_agent = PPO(
            "CnnPolicy", env, 
            policy_kwargs = policy_kwargs, verbose = 1
        )
        # save the randomly initialized AGENT18 for use as a baseline
        date_str = datetime.now().strftime('%Y%m%d_%H-%M')
        ppo_agent.save(f'baseline_ppo_agent_{args.dataset}_{date_str}_{args.num_epochs}_epochs_{args.batch_size}_batchsize')  # save untrained agent to use as a baseline

    # pretrain the PPO agent
    trained_agent = pretrain(ppo_agent, map_dataset, num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, dataset=args.dataset)
    print("trained agent!")

    # save the pretrained PPO agent
    date_str = datetime.now().strftime('%Y%m%d_%H-%M')
    trained_agent.save(f'pretrained_ppo_agent_{args.dataset}_{date_str}_{args.num_epochs}_epochs_{args.batch_size}_batchsize')
