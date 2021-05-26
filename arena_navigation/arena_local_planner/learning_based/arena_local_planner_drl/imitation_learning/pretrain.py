import os
import rospy
import rospkg
from datetime import datetime

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import random_split

from stable_baselines3 import PPO

from rl_agent.envs.flatland_gym_env import FlatlandEnv
from dataset import EpisodeDataset, MapDataset

from tensorboardX import SummaryWriter

# set default hyperparameters

#def pretrain(network, map_dataset, batch_size, optimizer):
#    # the map data_set is a list of EpisodeDatasets
#    # each EpisodeDatasets is a list of tuples: (observations, actions)
#
#    loss_fn = nn.MSELoss() # use mean-squared error loss for regression
#    
#    def train():
#        network.train()
#        # train and test sets should come from the same episode/path!
#        # iterate over the episodes in the map dataset
#        for episode in map_dataset:
#            # do 70% train, 30% test split for the episode  #TODO is there a scikitlearn function for this?
#            training_set_size = int(0.7*len(episode))
#            test_set_size = len(episode) - training_set_size
#
#            training_set, test_set = random_split(episode, [training_set_size, test_set_size])
#
#            train_loader = th.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)  #TODO how do these work under the hood?
#            test_loader = th.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)  #TODO this goes in the test() function
#
#            # iterate over the timesteps of the episode:
#            for batch, (observation, action) in enumerate(train_loader):
#                optimizer.zero_grad()
#                action_network = network(observation)  # run forward pass  #TODO check number of outputs
#                loss = loss_fn(action_network, action)  # compute loss
#                loss.backward()  # backpropagate loss
#                optimizer.step()  # adjust network parameters using the losses  #TODO check this
#
#    # initialize optimizer
#    # initialized learning rate scheduler
#
#    # call train() and test() repeatedly
#    # see ClipAssist training script
#    
#    # return trained network
#    pass

def pretrain(agent, map_dataset, num_epochs=1, batch_size=32, gamma=0.7, learning_rate=1.0):
    # the map data_set is a list of EpisodeDatasets
    # each EpisodeDatasets is a list of tuples: (observations, actions)

    #TODO see ClipAssist training script
    #TODO add random seed for reproducibility?
    #TODO add TensorBoard logger!
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
    optimizer = optim.Adadelta(network.parameters(), lr=learning_rate)  #TODO experiment with SGD, RMSprop and Adagrad
    # initialize learning rate scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)  #TODO understand hyperparameters step_size, gamma

    running_training_loss = 0
    running_test_loss = 0
    training_step_counter = 0
    test_step_counter = 0
    for epoch in range(num_epochs):
        # train and test sets should come from the same episode/path!
        # iterate over the episodes in the map dataset
        episode_counter = 1
        for episode in map_dataset:
            #print(f"training on episode {episode_counter}/{len(map_dataset)}")
            # do 70% train, 30% test split for the episode  #TODO is there a scikitlearn function for this?
            training_set_size = int(0.7*len(episode))
            test_set_size = len(episode) - training_set_size

            training_set, test_set = random_split(episode, [training_set_size, test_set_size])

            train_loader = th.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)  #TODO how do these work under the hood?
            test_loader = th.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

            # iterate over the timesteps of the training set:
            network.train()
            for batch, (observation, action) in enumerate(train_loader):
                optimizer.zero_grad()
                action_network, _, _ = network(observation)  # run forward pass  #TODO check number of outputs
                #print(f"action_network: {action_network}")
                action_network = action_network.double()  # MSELoss() expects inputs to be doubles
                #print(f"action_network.double(): {action_network}")
                loss = loss_fn(action_network, action)  # compute loss
                loss.backward()  # backpropagate loss
                optimizer.step()  # adjust network parameters using the losses  #TODO check this

                running_training_loss += loss.item()  # keep running tally of training loss over the last 1000 iterations
                if training_step_counter % 1000 == 999:  # every 1000 iterations, log training loss
                    writer.add_scalar('running training loss', running_training_loss/1000, training_step_counter)
                    running_training_loss = 0
                training_step_counter += 1
            #average_train_loss = train_loss / len(training_set)
            #print(f"epoch: {epoch} | episode {episode_counter}/{len(map_dataset)} | average training set loss: {average_train_loss}")
            
            # iterate over the timesteps of the test set to get the average test set loss
            network.eval()  #TODO don't update gradients? Need both eval and no_grad?
            with th.no_grad():  # don't update gradients!
                for batch, (observation, action) in enumerate(test_loader):
                    action_network, _, _ = network(observation)
                    action_network = action_network.double()
                    running_test_loss += loss_fn(action_network, action).item()  #TODO check me
                    
                    if test_step_counter % 1000 == 999:
                        writer.add_scalar('running test loss', running_test_loss/1000, test_step_counter)
                        running_test_loss = 0
                    test_step_counter += 1
                #average_loss = loss / len(test_set)
                #print(f"epoch: {epoch} | episode {episode_counter}/{len(map_dataset)} | average test set loss: {average_loss}")

            episode_counter += 1
        scheduler.step()
    
    agent.policy = network  # overwrite agent's network with the new pretrained one
    
    #writer.add_graph(network)  #TODO pass an observation (mini-batch) to add_graph() 
    #File "/home/michael/python_env/rosnav/lib/python3.6/site-packages/torch/utils/tensorboard/__init__.py", line 4, in <module>
    #raise ImportError('TensorBoard logging requires TensorBoard version 1.15 or above')
    
    writer.close()
    return agent

def run_agent(env, model, steps=500):
    obs = env.reset()

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if done:
            obs = env.reset()

        #time.sleep(0.1)

if __name__ == '__main__':  
    # create map dataset(s)
    map_dataset = MapDataset('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/output/map_empty_small/')
    scenario = '/home/michael/catkin_ws/src/arena-rosnav/gui/eval_scenarios/0_done_json_files/eval2_with_map_empty_small/07.0__2_static_2_dynamic_obs_robot_pos1.json'
    models_folder_path = rospkg.RosPack().get_path('simulator_setup')
    arena_local_planner_drl_folder_path = rospkg.RosPack().get_path('arena_local_planner_drl')

    ns = ''

    env = FlatlandEnv(ns=ns, PATHS={'robot_setting': os.path.join(models_folder_path, 'robot', 'myrobot.model.yaml'), 'robot_as': os.path.join(arena_local_planner_drl_folder_path,
                               'configs', 'default_settings.yaml'), "model": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/rule_00",
                               "scenerios_json_path": scenario,
                               "curriculum": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/configs/training_curriculum.yaml"},
                               reward_fnc="rule_00", is_action_space_discrete=False, debug=False, train_mode=True, max_steps_per_episode=100,
                               safe_dist=None, curr_stage=1
                  )

    # create map datasets
    map_dataset = MapDataset('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/output/map_empty_small')

    #TODO create PPO agent
    ppo_agent = PPO('MlpPolicy', env, verbose=1)  #TODO verbose=1?
    date_str = datetime.now().strftime('%Y%m%d_%H-%M')
    ppo_agent.save(f'baseline_ppo_agent_{date_str}')  # save untrained agent to use as a baseline

    #call pretrain() function
    trained_agent = pretrain(ppo_agent, map_dataset, num_epochs=500)
    print("trained agent!")
    #TODO return value: model. save as pth file for evaluation, running OR save the ppo agent as done above
    date_str = datetime.now().strftime('%Y%m%d_%H-%M')
    trained_agent.save(f'pretrained_ppo_agent_{date_str}')
