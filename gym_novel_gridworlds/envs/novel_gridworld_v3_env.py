# Author: Gyan Tatiya
# Email: Gyan.Tatiya@tufts.edu
'''
----------------------------------------
## Novelty No.3
## gridworld-v3: catastrophe--the recipe is changed and the agent needs to get more logs/stones/rubber
## tree, rubber_tree, rock, crafting_table, hard_crafting_table
## recipe: 5 tree + 2stone + 1rubber = 1 pogostick

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

class NovelGridworldV3Env(NovelGridworldV0Env):
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

        self.env_name = 'NovelGridworld-v3'
        self.items = ['wall', 'tree', 'rock', 'rubber_tree', 'crafting_table', 'pogo_stick', 'hard_crafting_table']
        self.items_id = self.set_items_id(self.items)
        self.items_quantity = {'tree': 6, 'rock': 2, 'rubber_tree' : 1, 'crafting_table': 1, 'pogo_stick': 0, 'hard_crafting_table': 1,}
        self.inventory_items_quantity = OrderedDict({item: 0 for item in self.items})
        self.initial_inventory = OrderedDict({item: 0 for item in self.items}) # all items to zero

        self.items_lidar = ['wall', 'tree', 'rock', 'rubber_tree', 'crafting_table', 'hard_crafting_table']
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

    def step(self, action):
        """
        Actions: {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Break', 4:'Craft'}
        """

        self.last_action = action
        r, c = self.agent_location

        done = False
        reward = -1  # default reward

        if action == 0: # forward
            if self.agent_facing_str == 'NORTH' and self.map[r - 1][c] == 0:
                self.agent_location = (r - 1, c)
            elif self.agent_facing_str == 'SOUTH' and self.map[r + 1][c] == 0:
                self.agent_location = (r + 1, c)
            elif self.agent_facing_str == 'WEST' and self.map[r][c - 1] == 0:
                self.agent_location = (r, c - 1)
            elif self.agent_facing_str == 'EAST' and self.map[r][c + 1] == 0:
                self.agent_location = (r, c + 1)

        elif action == 1: # left
            if self.agent_facing_str == 'NORTH':
                self.set_agent_facing('WEST')
            elif self.agent_facing_str == 'SOUTH':
                self.set_agent_facing('EAST')
            elif self.agent_facing_str == 'WEST':
                self.set_agent_facing('SOUTH')
            elif self.agent_facing_str == 'EAST':
                self.set_agent_facing('NORTH')

        elif action == 2:# Right
            if self.agent_facing_str == 'NORTH':
                self.set_agent_facing('EAST')
            elif self.agent_facing_str == 'SOUTH':
                self.set_agent_facing('WEST')
            elif self.agent_facing_str == 'WEST':
                self.set_agent_facing('NORTH')
            elif self.agent_facing_str == 'EAST':
                self.set_agent_facing('SOUTH')
        
        elif action == 3: # Break
            self.update_block_in_front()
            # If block in front is not air and wall (and crafting table), place the block in front in inventory
            if self.block_in_front_str == 'tree' or self.block_in_front_str == 'rock' or self.block_in_front_str == 'rubber_tree':
                block_r, block_c = self.block_in_front_location
                self.map[block_r][block_c] = 0
                if (self.block_in_front_str == 'tree' and self.inventory_items_quantity['tree'] <= 2) or (self.block_in_front_str == 'rock' and self.inventory_items_quantity['rock'] <= 1) or (self.block_in_front_str == 'rubber_tree' and self.inventory_items_quantity['rubber_tree'] <= 0):
                    reward = self.reward_break
                else:
                    reward = 0
                self.inventory_items_quantity[self.block_in_front_str] += 1

        elif action == 4: # craft
            self.update_block_in_front()
            # Catastrophic novelty - 'crafting_table' no longer actionable,
            # 'hard_crafting_table' has harder recipe
            if self.block_in_front_str == 'hard_crafting_table':
                if self.inventory_items_quantity['tree'] >= 5 and self.inventory_items_quantity['rock'] >= 2 and self.inventory_items_quantity['rubber_tree'] >= 1 : # recipe: 5logs+2stone+1rubber = 1 pogostick
                    self.inventory_items_quantity['pogo_stick'] += 1
                    done = True
                    reward = self.reward_done

        # Update after each step
        observation = self.get_observation()
        self.update_block_in_front()

        info = {}

        # Update after each step
        self.step_count += 1
        self.last_reward = reward
        self.last_done = done

        # if done == False and self.step_count == self.episode_timesteps:
        #     done = True

        return observation, reward, done, info
