'''
Adapted from the repository https://github.com/gtatiya/gym-novel-gridworlds
https://github.com/gtatiya/gym-novel-gridworlds/blob/master/tests/keyboard_interface.py

A keyboard interface to test the environment 
and play yourself!

NOTE: To run this you need to be the administrator
$ sudo python keyboard_interface.py

For questions contact shivam.goel@tufts.edu

'''

import os
import time

import gym
import gym_novel_gridworlds
from constants import env_key

import keyboard
import numpy as np
import matplotlib.image as mpimg


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

final_status = False # If True, reward shaping present in the task.


def print_play_keys(action_str):
    print("Press a key to play: ")
    for key, key_id in KEY_ACTION_DICT.items():
        print(key, ": ", action_str[key_id])


def get_action_from_keyboard():
    while True:
        key_pressed = keyboard.read_key()
        # return index of action if valid key is pressed
        if key_pressed:
            if key_pressed in KEY_ACTION_DICT:
                return KEY_ACTION_DICT[key_pressed]
            elif key_pressed == "esc":
                print("You pressed esc, exiting!!")
                break
            else:
                print("You pressed wrong key. Press Esc key to exit, OR:")
                print_play_keys(env.action_str)


def fix_item_location(item, location):
    result = np.where(env.map == env.items_id[item])
    if len(result) > 0:
        r, c = result[0][0], result[1][0]
        env.map[r][c] = 0
        env.map[location[0]][location[1]] = env.items_id[item]
    else:
        env.map[location[0]][location[1]] = env.items_id[item]

env_id = 'NovelGridworld-v0'
env = gym.make(env_id, map_width = width, map_height = height, items_quantity = {'tree': no_trees, 'rock': no_rocks, 'rubber_tree':no_rubber_tree,'crafting_table': crafting_table, 'pogo_stick':0},
    initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks,'rubber_tree': starting_rubber_trees, 'crafting_table': 0, 'pogo_stick':0}, goal_env = type_of_env, is_final = final_status)
env = gym.make(env_id)
# TODO - why do we reset the env with no parameters after making it the first time?

# NOTE - Testing other environments
# env_id = 'NovelGridworld-v3'
# env = gym.make(env_id, map_width = width, map_height = height, 
#                 goal_env = type_of_env, is_final = final_status)

# wrappers
# env = SaveTrajectories(env, save_path="saved_trajectories")

# observation_wrappers
# env = LidarInFront(env)
# env = AgentMap(env)

KEY_ACTION_DICT = env_key[env_id]

# # novelty_wrappers
# novelty_name = 'axetobreak'  # 'axe', 'axetobreak, 'fence'
# level, difficulty = 1, 'hard'  # easy, medium, hard
# if level == 1:
#     if difficulty == 'easy':
#         if novelty_name == 'axe':
#             env = Level1AxeEasy(env)
#             KEY_ACTION_DICT.update({"f": len(KEY_ACTION_DICT)})  # Select_axe
#         elif novelty_name == 'axetobreak':
#             env = Level1AxetoBreakEasy(env)
#             KEY_ACTION_DICT.update({"f": len(KEY_ACTION_DICT)})  # Select_axe
#         elif novelty_name == 'fence':
#             env = Level1Fence(env, difficulty)
#     elif difficulty == 'medium':
#         if novelty_name == 'axe':
#             env = Level1AxeMedium(env)
#             KEY_ACTION_DICT.update({"f": len(KEY_ACTION_DICT)})  # Select_axe
#         elif novelty_name == 'axetobreak':
#             env = Level1AxetoBreakMedium(env)
#             KEY_ACTION_DICT.update({"f": len(KEY_ACTION_DICT)})  # Select_axe
#         elif novelty_name == 'fence':
#             env = Level1Fence(env, difficulty)
#     elif difficulty == 'hard':
#         if novelty_name == 'axe':
#             env = Level1AxeHard(env)
#             KEY_ACTION_DICT.update({"5": len(KEY_ACTION_DICT)})  # Craft_axe
#             KEY_ACTION_DICT.update({"f": len(KEY_ACTION_DICT)})  # Select_axe
#         elif novelty_name == 'axetobreak':
#             env = Level1AxetoBreakHard(env)
#             KEY_ACTION_DICT.update({"5": len(KEY_ACTION_DICT)})  # Craft_axe
#             KEY_ACTION_DICT.update({"f": len(KEY_ACTION_DICT)})  # Select_axe
#         elif novelty_name == 'fence':
#             env = Level1Fence(env, difficulty)

# env = BlockItem(env)

# env.map_size = np.random.randint(low=10, high=20, size=1)[0]
# fix_item_location('crafting_table', (3, 2))

obs = env.reset()
env.render()
for i in range(100):
    print_play_keys(env.action_str)
    action = get_action_from_keyboard()  # take action from keyboard
    observation, reward, done, info = env.step(action)

    print("action: ", action, env.action_str[action])
    print("Step: " + str(i) + ", reward: ", reward)
    print("observation: ", len(observation), observation)

    print("inventory_items_quantity: ", len(env.inventory_items_quantity), env.inventory_items_quantity)
    print("items_id: ", len(env.items_id), env.items_id)

    try:
        print("step_cost, message: ", info['step_cost'], info['message'])
        print("selected_item: ", env.selected_item)
    except:
        pass

    time.sleep(0.2)
    print("")

    if i == 5:
        # env.remap_action()
        # print("action_str: ", env.action_str)
        # env.add_new_items({'rock': 3, 'axe': 1})
        # env.block_item(item_to_block='crafting_table', item_to_block_from='tree_log')
        pass

    env.render()
    if done:
        print("Finished after " + str(i) + " timesteps\n")
        time.sleep(2)
        obs = env.reset()
        env.render()

# env.save()
env.close()
