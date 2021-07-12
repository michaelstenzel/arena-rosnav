from dataset import EpisodeDataset, MapDataset

map_dataset = MapDataset('/home/michael/catkin_ws/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/imitation_learning/output/')

datapoints = 0
episodes = 0
for episode in map_dataset:
    episodes += 1
    datapoints += len(episode)

print(f"this directory has {datapoints} datapoints in {episodes} episodes")
