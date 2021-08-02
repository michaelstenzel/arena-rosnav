import os
import rospkg
import time

import numpy as np

from rl_agent.envs.flatland_gym_env import FlatlandEnv

import getch

# create map dataset(s)
models_folder_path = rospkg.RosPack().get_path('simulator_setup')
arena_local_planner_drl_folder_path = rospkg.RosPack().get_path('arena_local_planner_drl')

ns = ''

env = FlatlandEnv(ns=ns, PATHS={'robot_setting': os.path.join(models_folder_path, 'robot', 'myrobot.model.yaml'), 'robot_as': os.path.join(arena_local_planner_drl_folder_path,
                            'configs', 'default_settings.yaml'), "model": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/rule_00",
                            "curriculum": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/configs/training_curriculum.yaml"},
                            reward_fnc="rule_00", is_action_space_discrete=False, debug=False, train_mode=True, max_steps_per_episode=650,
                            safe_dist=None, curr_stage=3,
                            move_base_simple=False
                )

INCREMENT = 0.5

def print_action():
    os.system("clear")  # clear command line output
    print(f""" Action: {action}

     Key mapping:
     w: increase linear speed
     s: decrease linear speed
     a: increase angular speed (counter clockwise)
     d: decrease angular speed (clockwise)

     k: confirm action
     """)

def adjust_action(key):
    if key == "w":
        action[0] += INCREMENT
    if key == "s":
        action[0] -= INCREMENT
    if key == "a":
        action[1] += INCREMENT
    if key == "d":
        action[1] -= INCREMENT
    return

steps = 50000
obs = env.reset()

action = np.array([0.0, 0.0])
print_action()

for step in range(steps):
    key = "blocked"
    while key != "k":
        key = getch.getch()
        adjust_action(key)
        print_action()
    
    obs, rewards, done, info = env.step(action)
    print(f"rewards: {rewards}")
    env._steps_curr_episode += 1
    if done:
        obs = env.reset()

    time.sleep(0.1)