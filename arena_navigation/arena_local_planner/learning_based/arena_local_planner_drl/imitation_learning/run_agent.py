import os
import rospy
import rospkg
from datetime import datetime
import time

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import random_split

from stable_baselines3 import PPO

from rl_agent.envs.flatland_gym_env import FlatlandEnv
from dataset import EpisodeDataset, MapDataset

from tensorboardX import SummaryWriter

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
                            reward_fnc="rule_00", is_action_space_discrete=False, debug=False, train_mode=True, max_steps_per_episode=650,
                            safe_dist=None, curr_stage=1,
                            move_base_simple=False
                )

# create map datasets
map_dataset = MapDataset('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/output/map_empty_small')

#load PPO agent from zip
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/baseline_ppo_agent_20210526_17-26.zip', env)
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210526_17-33.zip', env)
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/AGENT_1_2021_04_02__22_03/best_model.zip', env)

# pretrained PPO with discrete actions:
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/MLP_B_128-64_P_64-64_V_64-64_relu_2021_03_29__14_16/best_model.zip', env)
#pretrained agent ~160 episodes (actions not synchronized)
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210527_20-03.zip', env)

#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210530_19-55.zip', env)
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/baseline_ppo_agent_20210530_19-49.zip', env)
# this is the first one used for continued DRL training:
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210530_23-21.zip', env)

# "naively oversampled" agent:
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/oversample_pretrained_ppo_agent_20210614_19-50.zip', env)

# trained with more data from the first 110 steps of an episode
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210615_22-05.zip', env)

# trained with ~40k steps + ~40k (only recording 90 steps per episode)
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210616_16-05.zip', env)

# trained with ~102k steps
ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210616_19-03.zip', env)
# trained with ~111k steps
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210616_21-59.zip', env)


steps = 50000
obs = env.reset()

for step in range(steps):
    action, _ = ppo_agent.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env._steps_curr_episode += 1
    if done:
        obs = env.reset()

    time.sleep(0.1)