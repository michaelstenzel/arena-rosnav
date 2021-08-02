import argparse
import time
from datetime import datetime
import os
import sys
from stable_baselines3 import A2C
from rl_agent.envs.flatland_gym_env import FlatlandEnv
#from flatland_gym_env_il import FlatlandEnvIL
from task_generator.tasks import get_predefined_task
import rospy
import rospkg
import numpy as np
from collections import OrderedDict
import subprocess


#TODO docstring
""" record_rollouts.py
"""

def clear_costmaps():
        bashCommand = "rosservice call /move_base/clear_costmaps"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        #self._service_clear_client()
        return output, error

def save_episode(observations, actions, map_name):
    # save observations and actions in an npz file:
    date_str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    path = str(f'./output/{map_name}_stage_3_4_dynamic_1_static/rollout_{date_str}')
    print(f'Saving episode to {path}')
    np.savez_compressed(
        path,
        observations = np.array(observations),
        actions = np.array(actions)
    )
    print('Done saving episode.')

#rospy.init_node("record_rollouts", anonymous=True, disable_signals=True)

ns = ''

parser = argparse.ArgumentParser(description='.')
parser.add_argument('-m', '--map_name', type=str, help='name of the map being recorded on')
parser.add_argument('-s', '--scenario', type=str, metavar="[scenario name]", default='/home/michael/catkin_ws/src/arena-rosnav/simulator_setup/scenarios/eval/obstacle_map1_obs20.json', help='path of scenario json file for deployment')
args = parser.parse_args()

#task = get_predefined_task(ns, mode="scenario", PATHS={"model": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/rule_00", "scenario": args.scenario})  # json scenario
#task = get_predefined_task(mode="random", PATHS={"scenerios_json_path": args.scenario})  # random scenario
#task = get_predefined_task(ns, mode="staged", start_stage = 1, PATHS={"scenerios_json_path": args.scenario, "curriculum": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/configs/training_curriculum.yaml"})  # staged scenarios
#task = get_predefined_task(ns, mode="staged", start_stage = 1, PATHS={"model": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/rule_00", "scenerios_json_path": args.scenario, "curriculum": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/configs/training_curriculum.yaml"})  # staged scenarios
#last one is the last used one

#print(task)

models_folder_path = rospkg.RosPack().get_path('simulator_setup')
arena_local_planner_drl_folder_path = rospkg.RosPack().get_path(
    'arena_local_planner_drl')

#env = FlatlandEnvIL(ns, task, os.path.join(models_folder_path, 'robot', 'myrobot.model.yaml'),
#                  os.path.join(arena_local_planner_drl_folder_path,
#                               'configs', 'default_settings.yaml'), "rule_00", False,
#                  )
#env = FlatlandEnv(ns, task, os.path.join(models_folder_path, 'robot', 'myrobot.model.yaml'),
#                  os.path.join(arena_local_planner_drl_folder_path,
#                               'configs', 'default_settings.yaml'), "rule_00", False,
#                  )  #TODO must use rule_00 for the reward calculator!

# set curr_stage=1 to select which stage is played during recording
env = FlatlandEnv(ns=ns, PATHS={'robot_setting': os.path.join(models_folder_path, 'robot', 'myrobot.model.yaml'), 'robot_as': os.path.join(arena_local_planner_drl_folder_path,
                               'configs', 'default_settings.yaml'), "model": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/rule_00",
                               "scenerios_json_path": args.scenario,
                               "curriculum": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/configs/training_curriculum.yaml"},
                               reward_fnc="rule_04", is_action_space_discrete=False, debug=False, train_mode=True, max_steps_per_episode=600,
                               safe_dist=None, curr_stage=3,
                               move_base_simple=True
                  )  # must set a reward_fnc for the reward calculator, it will return if the episode is done and why

print(f"env: {env}")

#rospy.init_node("record_rollouts", anonymous=True, disable_signals=True)

obs = env.reset()
episode_observations = []
episode_actions = []
while(True):
    #time.sleep(0.01)
    merged_obs, obs_dict, action = env.observation_collector.get_observations_and_action()

    reward, reward_info = env.reward_calculator.get_reward(
            obs_dict['laser_scan'], obs_dict['goal_in_robot_frame'])
    
    done, info = env.check_if_done(reward_info)

    if done:
        print(f"done info: {info}")
        if info['done_reason'] == 1:
            # if the episode is done because the robot collided with an obstacle, ignore this episode
            # reduce repeat count by 1 and start again
            print('collision')
            #task._num_repeats_curr_scene -= 1
            #env.task._num_repeats_curr_scene -= 1  #TODO check me! I think this only exists for ScenarioTasks! May not be relevant with new file structure anyway (1 file per path)
            #time.sleep(1.0)
            
            episode_observations = []
            episode_actions = []

            #env.task.next_stage(False)
            clear_costmaps()
            env.reset()
            clear_costmaps()
            #time.sleep(2.0)
        else:
            if action is not None:
                episode_observations.append(merged_obs)
                episode_actions.append(action)
            save_episode(episode_observations, episode_actions, args.map_name)
            episode_observations = []
            episode_actions = []
            # try resetting the environment: this will either reset the obstacles and robot and start another episode for recording
            # or it will end the recording because all scenarios have been run their maximum number of times
            try:
                #env.task.next_stage(False)
                clear_costmaps()
                env.reset()
                clear_costmaps()
            except Exception as e:
                print(e)
                print('All scenarios have been evaluated!')
                break
    else:
        # if the episode is not done, save this timesteps's observations and actions to the arrays and continue the episode
        if action is not None:
            episode_observations.append(merged_obs)
            episode_actions.append(action)
    #env._steps_curr_episode += 1  # increase step count to enforce maximum number of steps per episode
