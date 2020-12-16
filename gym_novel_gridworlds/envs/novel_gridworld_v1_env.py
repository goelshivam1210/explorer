# Author: Gyan Tatiya
# Email: Gyan.Tatiya@tufts.edu
'''
----------------------------------------
## Novelty No.1
## gridworld-v1: add fence to the env, the agent has to bypass it.
## Trees, Rubber_tree, rock, crafting_table, fence
## recipe: 3logs+2stone+1rubber = 1 pogostick



### Author: Shivam Goel
### Email: shivam.goel@tufts.edu
----------------------------------------
### Author: Michael Kotlik
### Email: michael.kotlik@tufts.edu
----------------------------------------

'''
import math
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .novel_gridworld_v0_env import NovelGridworldV0Env

class NovelGridworldV1Env(NovelGridworldV0Env):
    # metadata = {'render.modes': ['human']}
    """
    Goal:
        Navigation if goal_env = 0
        Breaking if goal_env = 1
        Crafting if goal_env = 2
    State: lidar sensor (8 beams) + inventory_items_quantity + block_in_front
    Action: {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Break', 4: 'Crafting'}
    """

    def __init__(self, map_width=None, map_height=None, items_id=None, items_quantity=None, initial_inventory = None, goal_env = None, is_final = False):
        super().__init__(map_width = map_width, map_height = map_height, goal_env = goal_env, is_final = is_final)

        self.env_name = 'NovelGridworld-v1'
        self.items = ['wall', 'tree', 'rock', 'rubber_tree', 'crafting_table', 'pogo_stick', 'fence']
        self.items_id = self.set_items_id(self.items)
        self.items_quantity = {'tree': 6, 'rock': 2, 'rubber_tree' : 1, 'crafting_table': 1, 'pogo_stick': 0, 'fence': 3,}
        self.inventory_items_quantity = OrderedDict({item: 0 for item in self.items})
        self.initial_inventory = OrderedDict({item: 0 for item in self.items}) # all items to zero

        self.items_lidar = ['wall', 'tree', 'rock', 'rubber_tree', 'crafting_table', 'fence']
        self.items_id_lidar = self.set_items_id(self.items_lidar)

        # NOTE - changed to account for block_type_vector
        self.low = np.array([0] * (len(self.items_lidar) * self.num_beams) + [0] * len(self.inventory_items_quantity) +
                            [0] * len(self.generate_block_type_vector()))
        self.high = np.array([self.max_beam_range] * (len(self.items_lidar) * self.num_beams) + [6] * len(
            self.inventory_items_quantity) + [1] * len(self.generate_block_type_vector()))
        self.observation_space = spaces.Box(self.low, self.high, dtype=int)

        if map_width is not None:
            self.map_width = map_width
        if map_height is not None:
            self.map_height = map_height
        if items_id is not None:
            self.items_id = items_id
        if items_quantity is not None:
            self.items_quantity = items_quantity
        if goal_env is not None:
            self.goal_env = goal_env
        if initial_inventory is not None:
            self.initial_inventory = initial_inventory
        if is_final == True:
            self.reward_break = +50
