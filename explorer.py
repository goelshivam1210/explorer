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

"""

from RegularPolicyGradient import RegularPolicyGradient
import numpy as np

import spacy
import en_core_web_lg

# def get_input_size(env):
#     # size is # total lidar beams + # inventory spots + # block_in_front spots
#     return env.num_beams * len(env.items_lidar) + \
#            len(env.inventory_items_quantity) + \
#            len(env.items)

def get_most_similar_item(item, old_items):
    # given name of new item, and list of names of old items, returns name of the most
    # similar old item
    nlp = en_core_web_lg.load()
    item_token = nlp(item.lower().replace('_', ' '))
    similarities = []
    for old_item in old_items:
        similarities.append(
            item_token.similarity(nlp(old_item.lower().replace('_', ' ')))
            )

    return old_items[similarities.index(max(similarities))]

def generate_adaptive_agent(old_env, new_env, old_agent, parameter_list,
                            copy_similar_weights=False):
    """generate_adaptive_agent produces a new agent (RegularPolicyGradient object),
        given an old environment (gym_novel_gridworld), a new environment, an old agent,
        a list of parameters to pass to the new RegularPolicyGradient agent, along with
        variables specifying the kind of weights to use for expanding the new agent's
        neural network and exploration mechanics.

        Meant to be called after the old agent is trained on a the old environment for
        some successful episodes, generating succesful trajectories that will be copied
        to the new agent and may be used for clever exploration. It is assumed that
        the environment will contain a new object. Note, it is required that items
        and items_lidar lists in old_env and new_env are identical in order, with the
        exception being that the new item is listed last in new_env.items and
        new_env.items_lidar.

        It is thus assumed that the feature vector (observation) returned by new_env
        will have an extra 10 elements corresponding to the new object. The policy
        gradient network in the new agent will be expanded, adding new weights between
        the inputs and the hidden layer

        If copy_similar_weights = False or 0, the weights will be assigned randomly
        using Xavier initialization.

        If copy_similar weights = True or 1, semantic similarity will be used to find
        the object that existed in the old environment with the name most similar to
        the novel object. The weights on the connecetions between the input nodes
        corresponding to that object and the hidden layer will then be used for the
        new object.

    Args:
        old_env ([type]): [description]
        new_env ([type]): [description]
        old_agent ([type]): [description]
        parameter_list ([type]): [description]
        model_file ([type]): [description]
        copy_similar_weights (bool, optional): [description]. Defaults to False.
    """
    # --- Get num new objects and check correct feature vector size in new_env --- #
    new_items = list(set(new_env.items) - set(old_env.items))  # names of new items
    old_input_size = len(old_env.high)
    new_input_size = len(new_env.high)

    # 10 features because: 8 for lidar, 1 for inventory, 1 for block_in_front
    num_new_inputs = len(new_items) * (10)

    if new_input_size != old_input_size + num_new_inputs:
        print(f"[generate_adaptive_agent] Error: expect 10 new features for each new "
              "object added to the environment, but input is of size {old_input_size} "
              "for old_env and {new_input_size} for new_env")
        return None
    
    # --- Create new agent object --- #
    parameter_list["input_size"] = new_input_size
    new_agent = RegularPolicyGradient(**parameter_list)
    new_agent.load_model_from_dict(old_agent._model)  # Load model from old agent

    # --- Expand network with appropriate weights --- #
    if not copy_similar_weights:  # Expand network with random weights
        new_agent.expand_random_weights(num_new_inputs)
    else:  # Copy weights for existing features
        for new_item in new_items:
            most_similar_old_item = get_most_similar_item(new_item, old_env.items)
            old_item_ind = old_env.items.index(most_similar_old_item)

            # NOTE - remove after satisfied with testing
            print(f"Item most similar to {new_item} is {most_similar_old_item}"
                  f", w/ index {old_item_ind}")

            total_num_beams = len(old_env.items_lidar) * old_env.num_beams
            input_inds = [old_item_ind + len(old_env.items_lidar) * beam_i
                        for beam_i in range(old_env.num_beams)]
            input_inds += [total_num_beams + old_item_ind]
            input_inds += [total_num_beams +
                        len(old_env.inventory_items_quantity) + old_item_ind]
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
            block_in_front_v = observation[total_num_beams + length_inventory:]
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
