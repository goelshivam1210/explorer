'''
All the parameters for the code train.py

For questions contact shivam.goel@tufts.edu

'''
import math
from numpy.core.fromnumeric import argsort


# agent variables
actionCnt = 5
D = 46 + 6 #8 beams x 5 items lidar + 6 inventory items + 6 items one-hot vector (5 block items+air)
NUM_HIDDEN = 30
GAMMA = 0.99
LEARNING_RATE = 1e-4
DECAY_RATE = 0.99
MAX_EPSILON = 0.1
MIN_EPSILON = 0.05
EXPLORATION_STOP = 100000
random_seed = 1
EPISODES = 300000
LAMBDA = -math.log(0.01) / EXPLORATION_STOP # speed of decay

# clever exploration variables
MAX_RHO = 0.4
MIN_RHO = 0.1

action_space = ['W','A','D','U','C']

# environment variables
env_id = 'NovelGridworld-v0'
env_id_0 = 'NovelGridworld-v0'
width = 12
height = 12
no_trees = 6
no_rocks = 2
no_rubber_tree = 1
crafting_table = 1
starting_trees = 0
starting_rocks = 0
starting_rubber_trees = 0
type_of_env = 2

# env1 variables
env_id_1 = 'NovelGridworld-v1'
no_fence = 3
starting_fence = 0

# env2 variables
env_id_2 = 'NovelGridworld-v2'
no_oak_tree = 1
starting_oak_tree = 0

# env3 variables
env_id_3 = 'NovelGridworld-v3'
hard_crafting_table=1

final_status = True # If True, reward shaping present in the task.
