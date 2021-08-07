# Imitation Learning

These docs are a collection of commands for each file/component used for behavior cloning.

More detailed explanations of command line arguments and functions can be found in the respective files.

## Recording rollouts from MPC (or other planner)
It's important to make sure that the subgoal topic is set correctly: MPC uses move_base_simple. For this purpose, there is a `move_base_simple` Flag in the constructor of this branch's version of FlatlandEnv. It will set which topic to get the subgoal from: False (default) for subgoal, True for move_base_simple/goal. This is also used
in the ObservationCollector and Task.
```
if move_base_simple:
    self._subgoal_sub = rospy.Subscriber(
        f'{self.ns_prefix}move_base_simple/goal', PoseStamped, self.callback_subgoal)  #this is used when recording demonstrations from MPC
    else:
        self._subgoal_sub = rospy.Subscriber(
            f'{self.ns_prefix}subgoal', PoseStamped, self.callback_subgoal)
```

This is set accordigly in `record_rollouts.py` the file which triggers random tasks and records and records rollouts to disk.

Commands:

1. roslauch:

    `roslaunch arena_bringup start_arena_flatland.launch disable_scenario:="false" map_file:="map_small" local_planner:=mpc train_mode:="true"`

    Change the map_file to the map you want to record on.

2. record_rollouts.py

    `python record_rollouts.py -map_name 'map_small' -stage 1`

    If you're recording on a different map, change it here too (is relevant for making sure the rollouts are saved in the right directory). Change stage to determine the number of dynamic obstacles in the map.
    The relevant training curriculum is set when instantiating FlatlandEnv.


Known issues: sometimes the robot will be spawned too close to a wall or other obstacles, which causes problems for the higher planners. This will add to the total recording time and usually leads to a timeout for the episode. Sometimes the simulation and recording script must be restarted.

## Recording from a human expert

Steer the robot using the keyboard. The actions and observations will be recorded in a synchronized fashion.
The robot still uses continuous actions. A command-line interface in `human_expert.py` will prompt the user. w,a,s,d keys increment/decrement the linear and angular velocities. o resets both to 0. k confirms the action and triggers a step in the simulation with it (`env.step(action)`).
A guide to the possible actions for the human to take are printed at each step.

(N.B. the MPC planner will not be triggered, because move_base_simple is set to False in human_expert.py)

Recording **randomly generated scenarios**:

1. roslaunch:
    `roslaunch arena_bringup start_arena_flatland.launch disable_scenario:="false" map_file:="map4" local_planner:="mpc" train_mode:="true"`

    If you want to record on a different map, change it here.

2. pretrain.py
    `python human_expert.py -m "map4" -stage 1`

    Change the map here (relevant for saving rollouts in the right directory).
    Also change the stage to be recorded in.
    The relevant training curriculum is set when instantiating FlatlandEnv.


Recording **scenarios specified in a scenario.json file**:

1. roslaunch
    `roslaunch arena_bringup start_arena_flatland.launch disable_scenario:=false scenario_file:=blocked_single_corridor.json enable_pedsim:=true map_file:="map_small" train_mode:="true"`

    N.B. `enable_pedsim:=true`
    
    `scenario_file`: filename of a scenario.json located in `simulator_setup/scenarios`

    `map_file`: name of the map in which the scenario takes place/which was used when designing the scenario with the GUI

2. pretrain.py
    `python human_expert.py -m "blocked_single_corridor" -scenario "blocked_single_corridor.json"`

    `scenario`: the same scenario file as above

    `m` name of the map in which the scenario takes place/which was used when designing the scenario with the GUI


File can be extended relatively straightforwardly to recording from a DRL agent: load the DRL agent and use
```
action, _ = ppo_agent.predict(obs, deterministic=True)
obs, rewards, done, info = env.step(action)
```
instead of querying the human expert.

## Pretraining (behavior cloning)

Script for doing behavior cloning (supervised learning) on rollouts (recorded observation-action pairs), recorded from MPC or a human expert.

**To train an agent from scratch**:

1. roslaunch
    `roslaunch arena_bringup start_arena_flatland.launch disable_scenario:="false" map_file:="map4" local_planner:="mpc" train_mode:="true"`

    Change map_file to be trained on.

2. pretrain.py
    `python pretrain.py -e 5 -bs 15 -lr 1.0 -ds 'human_expert'`

    Options:

    `e`: number of episodes to run training script for. Default=5.

    `bs`: size of the mini-batches used during training Default=15.

    `lr`: initial learning rate. Default=1.0.

    `ds`: dataset. The source of the data (`human_expert` or `mpc`). Used both for instantiating the MapDataset (PyTorch Dataset object) and for constructing the path when saving agents. Default='human_expert`


**To continue training an agent, for example when taking a DRL-trained agent and fine-tuning it on data recorded for a specific scenario**:

1. roslaunch:
    `roslaunch arena_bringup start_arena_flatland.launch disable_scenario:="false" map_file:="map4" local_planner:="mpc" train_mode:="true"`

2. pretrain.py

    `python pretrain.py --load agents/AGENT_1_2021_04_02__22_03/best_model.zip`

    `load` Flag: path of model *zip* file relative to `/arena_local_planner_drl`


## Running a pretrained agent

Option 1: arena-rosnav's run_agent.py (in scripts/deployment). Runs agent in specified scenario file. Can handle agents which were trained on normalized observations.
1. roslaunch
    `roslaunch arena_bringup start_arena_flatland.launch map_file:="map1"  disable_scenario:="false" scenario_file:="eval/obstacle_map1_obs20.json"`

2. run_agent.py
    `python run_agent.py --load AGENT_20_2021_04_17__03_50 --scenario eval/obstacle_map1_obs20`

    `load`: name of agent in `/agents`

    `scenario`: path of scenario json (without json file extension) relative to `simulator_setup/scenarios`

Option 2: imitation_learning/run_agent.py. Generates random scenarios. Stage set inline in file (`curr_stage`). (**experimental, not meant for deployment**)
1. roslaunch
    `roslaunch arena_bringup start_arena_flatland.launch disable_scenario:="false" map_file:="map_small" local_planner:="mpc" train_mode:="false"`

2. run_agent.py
    `run_agent.py`

    Agent must be set by setting the absolute path to the zip file (e.g. best_model.zip) inline. Can't handle agents trained on normalized data.

When running a specific scenario generated using the GUI, use this command instead (needs pedsim):
    
    `roslaunch arena_bringup start_arena_flatland.launch disable_scenario:=false scenario_file:=blocked_single_corridor.json enable_pedsim:=true map_file:="map_small"`

## DRL Training

This section outlines the steps and commands that need to be executed to train a pretrained agent and its corresponding baseline agent.

### 0. Running long training jobs using Screen (OPTIONAL)

The training jobs can run up to several days, so it can be a good idea to run them using Screen. This way you can create a virtual session which will keep running on the server, even if the SSH connection is interrupted.

Alternatively, you can also use ```tmux```.

You will need two terminals: one for arena and one for the training script.

-Create screens:
```
screen -S simulation
screen -S training
```

Then run the training commands listed in section 1 below.

-To detach screens during training (for example if you want to power off your own machine), press ```CTRL-A + CTRL-D```

-To reatach screens to check progress, run these commands in terminals:
```
screen -r simulation
screen -r training
```

### 1. Launching a training session

### 1.1 First terminal
The first terminal (the simulation session, if using Screen) is needed to run arena.
Run these four commands:
```
source $HOME/.zshrc                         
source $HOME/catkin_ws/devel/setup.zsh    
workon rosnav
roslaunch arena_bringup start_training.launch train_mode:=true task_mode:=staged map_file:=map4 num_envs:=24
```

### 1.2 Second terminal 
A second terminal (the training session if using Screen) is needed to run the training script.
Run these four commands:
```
source $HOME/.zshrc                        
source $HOME/catkin_ws/devel/setup.zsh   
workon rosnav
roscd arena_local_planner_drl
```

Now, run one of the two commands below to start a training session:
```
python scripts/training/train_agent.py --load pretrained_ppo_human --n_envs 24 --eval_log --tb
```
```
python scripts/training/train_agent.py --load baseline_ppo_human --n_envs 24 --eval_log --tb
```

Setting ```--tb``` could throw a tensorboard error during training if there is a dependency mismatch and terminate the process. Remove ```--tb``` flag if this happens.

### 2. Ending a training session

When the training script is done, it will print the following information and then exit:
```
Time passed: {time in seconds}s
Training script will be terminated
```

Please make a note of this time taken.

The simulation will not terminate automatically, so it needs to be stopped with CTRL-C.