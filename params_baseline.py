'''
All the parameters for the code train.py

For questions contact shivam.goel@tufts.edu

'''
import math
from numpy.core.fromnumeric import argsort


# agent variables
actionCnt = 5
D = 46 + 6 #8 beams x 5 items lidar + 6 inventory items + 6 items one-hot vector (5 block items+air)
NUM_HIDDEN = 20
GAMMA = 0.99
LEARNING_RATE = 1e-4
DECAY_RATE = 0.99
MAX_EPSILON = 0.1
MIN_EPSILON = 0.05
EXPLORATION_STOP = 50000
random_seed = 1
EPISODES = 1000000
LAMBDA = -math.log(0.01) / EXPLORATION_STOP # speed of decay

action_space = ['W','A','D','U','C']

# environment variables
env_id = 'NovelGridworld-v0'
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

final_status = True # If True, reward shaping present in the task.