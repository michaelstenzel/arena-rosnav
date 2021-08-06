import os
import rospkg
import time

from stable_baselines3 import PPO

from rl_agent.envs.flatland_gym_env import FlatlandEnv

# create map dataset(s)
models_folder_path = rospkg.RosPack().get_path('simulator_setup')
arena_local_planner_drl_folder_path = rospkg.RosPack().get_path('arena_local_planner_drl')

ns = ''

env = FlatlandEnv(ns=ns, PATHS={'robot_setting': os.path.join(models_folder_path, 'robot', 'myrobot.model.yaml'), 'robot_as': os.path.join(arena_local_planner_drl_folder_path,
                            'configs', 'default_settings.yaml'), "model": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/rule_00",
                            "curriculum": "/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/configs/training_curriculum_map1small.yaml"},
                            reward_fnc="rule_00", is_action_space_discrete=False, debug=False, train_mode=True, max_steps_per_episode=650,
                            safe_dist=None, goal_radius=0.25, curr_stage=4,
                            move_base_simple=False
                )

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

# trained with ~102k steps - THIS IS THE LAST ONE I INVESTIGATED!!!
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210616_19-03.zip', env)
# trained with ~111k steps
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210616_21-59.zip', env)

# July 4th - fixed train/test split. 8 epochs@batch_size=32 - used this as my pretrained PPO agent
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210704_20-27.zip', env)

# corresponding baseline:
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/baseline_ppo_agent_20210704_20-22.zip', env)
# July 4th - fixed train/test split. 40 epochs@batch_size=32
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210704_21-01.zip', env)
# July 4th - fixed train/test split. 40 epochs@batch_size=8
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210704_22-20.zip', env)

##human expert ~43 episodes, first attempt, not well planned
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210720_23-32.zip', env)

# DRL agent recommended by TAL
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/rule_04/best_model.zip', env)
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/AGENT_20_2021_04_17__03_50/best_model.zip', env)

# human expert 57k timesteps, 384 episodes
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_20210727_11-36.zip', env)
# human expert 57k timesteps, 384 episodes - iterating over entire datasets (not episodes)
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_default-architecture_human_expert_20210730_16-29_10_epochs_15_batchsize_1.0_lr', env)

# human expert 57k timesteps - iterating over entire dataset, CNN agent 18 - 10 epochs
# this exhibited classic signs of distributional shift. Can turn quickly at start of episode, but if done imperfectly there are compounding errors.
# Positives: is finally interested in the goal. If if misses the goal, it will start circling around it.
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_18_human_expert_20210730_18-09_10_epochs_15_batchsize_1.0_lr', env)

# human expert 57k timesteps - iterating over entire dataset, CNN agent 18 - 5 epochs
# this is the best so far (July 30th, 8:28PM) - agent can competently turn and reach the goal!
# struggles with dynamic obstacles - can't avoid them if it's being approached by them. Sometimes avoid them if they aren't.
# Usually avoids static obstacles in the map, but every once in a while it will tunnel into a wall.
# This one was used for continued DRL:
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_18_human_expert_20210730_19-24_5_epochs_15_batchsize_1.0_lr', env)

# try running converted 5 epoch agent:
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_human.zip', env)
# try converted baseline 5 epoch agent:
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/baseline_ppo_human.zip', env)

# try running best_model with 80% success rate in stage 5, having passed stage 4 already:
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/best_model.zip', env)

# human expert 57k timesteps - iterating over entire dataset, CNN agent 18 - 4 epochs
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/pretrained_ppo_agent_18_human_expert_20210730_19-54_4_epochs_15_batchsize_1.0_lr', env)

# associated baseline to 5 epochs:
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/baseline_ppo_agent_18_human_expert_20210730_19-19_5_epochs_15_batchsize_1.0_lr', env)

# sanity check:
#ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/pretrained_ppo_human_expert/best_model.zip', env)

# fully DRL trained CNN agent 18 from repo - doesn't work at all?
ppo_agent = PPO.load('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/agents/AGENT_18_2021_04_11__13_54/best_model.zip', env)

steps = 50000
obs = env.reset()

for step in range(steps):
    os.system("clear")  # clear command line output
    action, _ = ppo_agent.predict(obs, deterministic=True)
    print(f'action: {action}')
    obs, rewards, done, info = env.step(action)
    print(f'rho: {obs[-2]}')
    print(f'theta: {obs[-1]}')
    env._steps_curr_episode += 1
    if done:
        obs = env.reset()

    time.sleep(0.1)