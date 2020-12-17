"""
The goal of this class is to find the state space 
initialize the new environment
modify the network
initialize the weights with similarity to the existing object weights

use sampled trajectories to generate optimal action set
use the optimal action set to perform reasoning over exploration over current 
observed states.

also add rewards based on developing agent to be more curious to even
visit novel states

run this to evaluate 
$ python explorer.py # need to add stuff

For questions contact
shivam.goel@tufts.edu
michael.kotliK@tufts.edu
# Michael add email here

"""

from RegularPolicyGradient import RegularPolicyGradient
import numpy as np

# NOTE - explanation of the layout of the features in the input vector
# self.low = np.array([0] * (len(self.items_lidar) * self.num_beams) + [0] * len(self.inventory_items_quantity))
# [0] * (5 * 8) + [0] * ()
# num_inputs <- initially set to 46 + 6 (for 5 objects)
# Corresponds to 8*5 (Lidar) + 5+1 inventory items (for pogo_stick) + 6 block_in_front (for air)
# within lidar_signals vector, grouped by angles
# So if there are 5 items, then first 5 indices correspond to beams for the 5 objects
# at angle 0, then the next 5 indices correspond to the next angle, and so on

"""
def get_observation(self):
    lidar_signals = self.get_lidarSignal()
    block_type_vector = self.generate_block_type_vector()
    observation = lidar_signals + [self.inventory_items_quantity[item] for item in
                                    sorted(self.inventory_items_quantity)]   
    observation = np.concatenate((observation, block_type_vector))
    return np.array(observation)
"""


def generate_expanded_agent(
    old_env, new_env, old_agent, parameter_list, model_file, copy_object_weights=None
):
    # NOTE - currently not using model_file, loading model directly from old agent
    # parameter_list should contain values for all the keys used to initialize agent
    # copy_object_weights should be the indices in the sorted arrays of the objects
    # whose connection weights should be copied

    new_items = list(set(new_env.items) - set(old_env.items))  # names of new items
    # new_input_size = len(new_env.high) # use once observation_space fixed
    num_new_inputs = len(new_items) * (
        10
    )  # 8 for lidar, 1 for inventory, 1 for block_in_front
    new_input_size = (
        new_env.num_beams * len(new_env.items_lidar)
        + len(new_env.inventory_items_quantity)
        + len(new_env.items)
    )  # size is # total lidar beams + # inventory spots + # block_in_front spots

    parameter_list["D"] = new_input_size  # Create new agent with new number of inputs
    new_agent = RegularPolicyGradient(**parameter_list)
    new_agent.load_model_from_dict(old_agent._model)  # Load model from old agent

    # TODO - figure out how to load model from file
    # new_agent.load_model(curriculum_no = 0, beam_no = 0, env_no = 1, ep_number=args['model'])

    if copy_object_weights is None:  # Expand network with random weights
        new_agent.expand_random_weights(num_new_inputs)
    else:  # Copy weights for existing features
        if len(copy_object_weights) != len(new_items):
            print(
                "[generate_expanded_agent] Error: length copy_input_weights must match"
                "number of new input nodes if copy_input_weights != None"
            )
            return None

        total_num_beams = len(old_env.items_lidar) * old_env.num_beams
        for object_ind in copy_object_weights:
            input_inds = [
                object_ind + len(old_env.items_lidar) * beam_i
                for beam_i in range(old_env.num_beams)
            ]
            input_inds += [total_num_beams + object_ind]
            input_inds += [
                total_num_beams + len(old_env.inventory_items_quantity) + object_ind
            ]
            new_agent.expand_copy_weights(input_inds)

    return new_agent


def get_optimal_actions(self, old_env, old_agent, similar_object_inds, metric="counts"):
    # similar_object_inds is a list of indices of objects of interest (within items array)
    # store list of actions taken in front of objects of interest, where
    # each entry is the action, number of times it was taken, and list of discounted
    # rewards received
    actions_taken = [[a_ind, 0, []] for a_ind in range(len(old_env.action_str))]

    total_num_beams = len(old_env.items_lidar) * old_env.num_beams
    length_inventory = len(old_env.inventory_items_quantity)
    # length_block_in_front = len(old_env.items)

    # populate action taken with counts and rewards
    for ep_ind in range(len(old_agent.succ_trajectories)):
        episode_SAR = old_agent.succ_trajectories[ep_ind]
        # 0 - observations, 1 - actions, 2 - discounted rewards
        for t in range(episode_SAR[0].shape[0]):
            observation = episode_SAR[0][t]
            block_in_front_v = observation[total_num_beams + length_inventory :]
            # check if in front of some block
            id_block_in_front = np.where(block_in_front_v == 1)
            if len(id_block_in_front[0]):
                id_bif = id_block_in_front[0][0]
                # check if block is one of interest
                if id_bif in similar_object_inds:
                    # get action and reward
                    action = episode_SAR[1][t]
                    reward = episode_SAR[2][t, 0]

                    # increase action count and append reward
                    actions_taken[action][1] += 1
                    actions_taken[action][2].append(reward)

    # rank optimal actions for each object based on times taken
    if metric == "counts":
        actions_taken_s = sorted(actions_taken, key=lambda x: x[1], reverse=True)
    # rank optimal actions for each object based on sum of discounted rewards
    elif metric == "reward":
        actions_taken_s = sorted(actions_taken, key=lambda x: sum(x[2]), reverse=True)
    else:
        print(
            "[get_optimal_actions] Error: only metrics 'counts' and 'reward' available"
        )

    optimal_actions = map(lambda x: x[0], actions_taken_s)
    return optimal_actions


# lists: items & items_lidar
# dict: items_id is a dict that's being used as if sorted, accessing keys and valyes
# dict: items_id_lidar has no functions depending on its ordering
# dict: inventory_items_quantity is being used sorted

# lidar is dependent on ordered items in self.items_lidar
# ex. ['wall', 'tree', 'rock', 'rubber_tree', 'crafting_table']
# inventory_items_quantity is dependent on ordered items in self.items
# ex. ['wall', 'tree', 'rock', 'rubber_tree', 'crafting_table', 'pogo_stick']
# block_type_vector is dependent on ordered items in self.items (replace 'pogo_stick' with 'air')
# ex. ['wall', 'tree', 'rock', 'rubber_tree', 'crafting_table', 'air']

# TODO - print out lengths of lidarSignals, inventory_items, block_in_front vector
# TODO - test after making changes


# TODO - need to prevent setting initial_inventory
# TODO - or give warning about order of keys

# NOTE - what should the exact details of the new tree novelty be? should we be
# adding the item such that it can be detected as the block in front and with LIDAR
# but in the inventory it's the same tree? That makes it harder to make explorer
# functions that generalize. Even with wall and fence it was assumed that there
# would be slots for that in the inventory but those slots would never be taken up.
# Maybe instead, oak_tree should be a separate item, but the number of tree + oak_tree
# together should be at least 3 to craft the pogo_stick
# TODO - change how Env2 works

# NOTE - we give reward for the first two trees broken

"""
self.items = ['wall', 'tree', 'rock', 'rubber_tree', 'crafting_table', 'pogo_stick']
self.items_id = self.set_items_id(self.items)
# items_quantity when the episode starts, do not include wall, quantity must be more than 0
self.items_quantity = {'tree': 6, 'rock': 2, 'rubber_tree' : 1, 'crafting_table': 1, 'pogo_stick': 0}
self.inventory_items_quantity = {item: 0 for item in self.items}
self.items_lidar = ['wall', 'crafting_table', 'tree', 'rock', 'rubber_tree']
self.items_id_lidar = self.set_items_id(self.items_lidar)
"""

"""
# If bean hit an object or wall
                if obj_id_rc != 0:
                    item = list(self.items_id.keys())[list(self.items_id.values()).index(obj_id_rc)]
                    if item in self.items_id_lidar:
                        obj_id_rc = self.items_id_lidar[item]
                        beam_signal[obj_id_rc - 1] = beam_range
"""

"""
observation = lidar_signals + [self.inventory_items_quantity[item] for item in
                                       sorted(self.inventory_items_quantity)]
"""

"""
# one-hot vector of the type of block in front of the agent
def generate_block_type_vector(self):
    # reset and create a new dictionary to replace pogostick with air
    items_dict_2 = copy.deepcopy(self.items_id)
    items_dict_2['air'] = items_dict_2.pop('pogo_stick') 
    ret = np.zeros((len(items_dict_2.keys())))# make a new array to return one-hot vector
    for k,v in items_dict_2.items():
        if self.block_in_front_str == k:
            ret[v-1] = 1
    return ret
"""

"""
def set_items_id(self, items):
    items_id = {}
    for item in sorted(items):
        items_id[item] = len(items_id) + 1

    return items_id
"""

# NOTE - use OrderedDict instead