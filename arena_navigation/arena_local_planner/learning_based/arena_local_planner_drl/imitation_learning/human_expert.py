import argparse
from datetime import datetime
import os
from rl_agent.envs.flatland_gym_env import FlatlandEnv
import rospkg
import numpy as np
from pathlib import Path

import getch


#TODO docstring
""" human_expert.py
"""

def save_episode(observations, actions, rewards, dones, infos):
    # save observations and actions in an npz file:
    directory = f'./output/human_expert/{args.map_name}_stage_{args.stage}'  # directory to store the rollouts in
    Path(directory).mkdir(parents=True, exist_ok=True)  # make directory if it doesn't exist yet
    date_str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    path_to_file = f'{directory}/rollout_{date_str}'
    print(f'Saving episode to {path_to_file}')
    np.savez_compressed(
        path_to_file,
        observations = np.array(observations),
        actions = np.array(actions),
        rewards = np.array(rewards),
        dones = np.array(dones),
        infos = np.array(infos)
    )
    print('Done saving episode.')

ns = ''

parser = argparse.ArgumentParser(description='.')
parser.add_argument('-m', '--map_name', type=str, help='name of the map being recorded on', required=True, default="map_small")
parser.add_argument('-scenario', '--scenario', type=str, metavar="[scenario name]", default='', help='name with .json of the scenario json file in /simulator_setup/scenarios')
parser.add_argument('-stage', '--stage', type=int, metavar="[current stage]", default=1, help='stage to start the simulation with')
args = parser.parse_args()

if args.scenario != '':
    task_mode = 'scenario'
else:
    task_mode = 'staged'

models_folder_path = rospkg.RosPack().get_path('simulator_setup')
arena_local_planner_drl_folder_path = rospkg.RosPack().get_path(
    'arena_local_planner_drl')

# relevant parameters:
# task_mode (staged for random scenarios, scenario for a predefined scenario)
# curr_stage: current stage in "curriculum"
# scenario: name (WITH file extension) of scenario file
# max_steps_per_episode: maximum number of steps to record. Script will save episodes if maximum number of steps is reached!
env = FlatlandEnv(ns=ns, PATHS={'robot_setting': os.path.join(models_folder_path, 'robot', 'myrobot.model.yaml'), 'robot_as': os.path.join(arena_local_planner_drl_folder_path,
                               'configs', 'default_settings.yaml'), "model": os.path.join(arena_local_planner_drl_folder_path, 'agents', 'rule_04'),
                               "scenario": os.path.join(models_folder_path, 'scenarios', args.scenario),
                               "curriculum": os.path.join(arena_local_planner_drl_folder_path, 'configs', 'training_curriculum_map1small.yaml')},
                               reward_fnc="rule_04", is_action_space_discrete=False, debug=False, train_mode=True, max_steps_per_episode=5000,
                               task_mode=task_mode,
                               safe_dist=None, curr_stage=args.stage,
                               move_base_simple=False
                  )  # must set a reward_fnc for the reward calculator, it will return if the episode is done and why

print(f"env: {env}")

# constants and functions for steering the agent
LINEAR_INCREMENT = 0.1
ANGULAR_INCREMENT = 0.5
MAX_LINEAR_VELOCITY = env.action_space.high[0]
MIN_LINEAR_VELOCITY = env.action_space.low[0]
MAX_ANGULAR_VELOCITY = env.action_space.high[1]
MIN_ANGULAR_VELOCITY = env.action_space.low[1]

def print_action():
    os.system("clear")  # clear command line output
    print(f'max linear: {MAX_LINEAR_VELOCITY}')
    print(f'min linear: {MIN_LINEAR_VELOCITY}')
    print(f'max angular: {MAX_ANGULAR_VELOCITY}')
    print(f'min angular: {MIN_ANGULAR_VELOCITY}')
    print(f""" Action: {action}

     Key mapping:
     w: increase linear speed
     s: decrease linear speed
     a: increase angular speed (counter clockwise)
     d: decrease angular speed (clockwise)

     o: reset to [0, 0]

     k: confirm action
     """)

def select_action():
    # let human expert adjust the speed and confirm when done. The new action will be stored in the global action variable.
    key = "blocked"
    while key != "k":
        key = getch.getch()
        adjust_action(key)
        print_action()

def adjust_action(key):
    if key == "w":
        action[0] += LINEAR_INCREMENT
        action[0] = np.clip(action[0], MIN_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY)
    if key == "s":
        action[0] -= LINEAR_INCREMENT
        action[0] = np.clip(action[0], MIN_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY)
    if key == "a":
        action[1] += ANGULAR_INCREMENT
        action[1] = np.clip(action[1], MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
    if key == "d":
        action[1] -= ANGULAR_INCREMENT
        action[1] = np.clip(action[1], MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
    if key == "o":
        action[0], action[1] = 0.0, 0.0
    return

# print initial action (standing still)
action = np.array([0.0, 0.0])
print_action()

# get first observation and action pair - these will not be recorded, since env.reset() cannot return reward info
obs = env.reset()
select_action()

# initialize lists to store observations and actions
episode_observations = []
episode_actions = []
episode_rewards = []
episode_dones = []
episode_infos = []

while(True):
    # take a step with the selected action and return data on the new state of the system
    obs, rewards, done, info = env.step(action)
    print(f'rho: {obs[-2]}')
    print(f'theta: {obs[-1]}')

    # let human expert adjust the speed and confirm when done. The new action will be stored in the global action variable.
    select_action()

    if done:
        print(f"done info: {info}")
        if info['done_reason'] == 1:
            # if the episode is done because the robot collided with an obstacle, ignore this episode
            # reset episode lists to empty lists
            print('collision')
            
            episode_observations = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_infos = []
            
            env.reset()
            select_action()
        else:
            # done but not crashed - ran out of timesteps or at the goal
            episode_observations.append(obs)
            episode_actions.append(np.array(action))
            episode_rewards.append(rewards)
            episode_dones.append(done)
            episode_infos.append(info)

            save_episode(episode_observations, episode_actions, episode_rewards, episode_dones, episode_infos)
            episode_observations = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_infos = []
            # try resetting the environment: this will either reset the obstacles and robot and start another episode for recording
            # or it will end the recording because all scenarios have been run their maximum number of times
            try:
                env.reset()
                select_action()
            except Exception as e:
                print(e)
                print('All scenarios have been evaluated!')
                break
    else:
        # if the episode is not done, save this timesteps's observations and actions to the arrays and continue the episode
        episode_observations.append(obs)
        episode_actions.append(np.array(action))
        episode_rewards.append(rewards)
        episode_dones.append(done)
        episode_infos.append(info)
    env._steps_curr_episode += 1  # increase step count to enforce maximum number of steps per episode
