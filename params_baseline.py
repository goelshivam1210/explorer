'''
All the parameters for the code train.py

For questions contact shivam.goel@tufts.edu

'''
import math

# agent variables
actionCnt = 5
D = 46 #8 beams x 5 items lidar + 6 inventory items
NUM_HIDDEN = 20
GAMMA = 0.99
LEARNING_RATE = 1e-4
DECAY_RATE = 0.99
MAX_EPSILON = 0.9
MIN_EPSILON = 0.05
EXPLORATION_STOP = 50000
random_seed = 1
EPISODES = 300000
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