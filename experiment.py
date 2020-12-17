
'''
This code trains the novelgridworlds environment task (novel_gridworld_v0_env.py)
using Regular Policy Gradient algorithm (RegularPolicyGradient.py)
Saves the models in results/<model_name>
Saves the csv(s) for plotting in plot/data/<train_results.csv>

run this to train 
$ python train.py -C <resume_training{True/False}> -M <model_episode_number> -E <episode_number> -R <render{True/False}> -P <print result every "X" episode>

For questions contact shivam.goel@tufts.edu

'''


import os
import csv
import math
import argparse

import gym
import numpy as np
import gym_novel_gridworlds


from RegularPolicyGradient import RegularPolicyGradient
from params_baseline import *
import matplotlib.pyplot as plt

def save_results (data, tag):
    os.makedirs("plot" + os.sep + "data", exist_ok=True)
    if tag == 'train_results':
        db_file_name = "plot/data/train_results.csv"
        with open(db_file_name, 'a') as f: # append to the file created
            writer = csv.writer(f)
            writer.writerow(data)


###### copy from explorer.py
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

    ############## change here ###############
    parameter_list["input_size"] = new_input_size  # Create new agent with new number of inputs
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

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-C", "--continue", default=False, help="Want to continue training or start from scratch", type=bool)
    ap.add_argument("-M", "--model", default= 'final', help="Episode number of the model you want to load")
    ap.add_argument("-E", "--episode", default= 0, help="Episode number from to resume(used for saving results)", type=int)
    ap.add_argument("-R","--render", default = False, help="Want to render or not(False or True)", type = bool)
    ap.add_argument("-P", "--print_every", default= 100, help="Number of epsiodes you want to print the results", type=int)
    ap.add_argument("-N", "--num_model", default= 4, help="Number of models want to save before final trained model. Used for evaluations", type=int)
    #################### change #################
    ap.add_argument("-G", "--gridworld", default= 0, help="new env to be loaded (1 or 2 or 3)", type=int) ### new added
    ap.add_argument("-W", "--weights", default= None, help="the weight input for the agent") ### new added
    

    ap.add_argument("-print_output", default="", help="print stuff")
    args = vars(ap.parse_args())

    # load the learning agent
    parameter_list = {
        "num_actions": actionCnt, 
        "input_size": D, 
        "hidden_layer_size": NUM_HIDDEN,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "decay_rate": DECAY_RATE,
        "greedy_e_epsilon": MAX_EPSILON,
        "random_seed": random_seed
    }

    agent = RegularPolicyGradient(**parameter_list)

    # agent = RegularPolicyGradient(actionCnt,D,NUM_HIDDEN,\
    #                  LEARNING_RATE,GAMMA,DECAY_RATE,\
    #                  MAX_EPSILON,random_seed)

    if args['continue'] == True:
        print ("LOADING model ....")
        agent.load_model(curriculum_no = 0, beam_no = 0, env_no = 1, ep_number=args['model'])
    # get the environment
    
    env = gym.make(env_id_0,\
                  map_width = width, map_height = height,\
                  items_quantity = {'tree': no_trees, 'rock': no_rocks, 'rubber_tree':no_rubber_tree,'crafting_table': crafting_table, 'pogo_stick':0},
                  initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks,'rubber_tree': starting_rubber_trees, 'crafting_table': 0, 'pogo_stick':0},\
                  goal_env = type_of_env, is_final = final_status)
    
    t_step = 0
    episode = args['episode']
    t_limit = 150
    reward_sum = 0
    reward_arr = []
    avg_reward = []
    done_arr = []
    env_flag = 0

    env.reset()

    while True:    
        # get obseration from sensor
        obs = env.get_observation()

        # set epsilon based on the decay rate
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA*episode)
        agent.set_explore_epsilon(epsilon)

        # act 
        a = agent.process_step(obs,True)

        new_obs, reward, done, info = env.step(a)
        # print ("observation = {} shape = {}".format(new_obs, new_obs.shape))
        # print ("reward = {}".format(reward))
        # print ("done = {}".format(done))
       
        if args['render'] == True:
            env.render()

        # give reward
        agent.give_reward(reward)
        reward_sum += reward
        
        t_step += 1
        
        if t_step > t_limit or done == True:
            # print every 100th episode results
            if (episode%int(args['print_every']) == 0):
                print("Episode--> {} Reward --> {} EPS --> {}".format(episode, reward_sum, np.round(agent._explore_eps, decimals = 2)))

            reward_arr.append(reward_sum)
            avg_reward.append(np.mean(reward_arr[-40:]))
    
            # done = True # commenting out because finish_episode now needs actual done
            t_step = 0
            agent.finish_episode(done)
        
            # update after every 10 episodes
            if episode % 10 == 0:
                agent.update_parameters()
        
            # reset environment
            episode += 1
            ## save the rewards for plotting
            data = [episode, reward_sum, agent._explore_eps]
            save_results(data, tag = 'train_results')

            env.reset()
            reward_sum = 0
            # save only 4 models
            if episode% (EPISODES/args['num_model']) == 0:
                agent.save_model(0,0,1,episode)
    
            # change to a new_env after 7000 trails
            if episode == 700:
                ######## change the file name here
                agent.save_model(0,0,1,'mid') # Harcoded for now

                ######## new env env[1]/env[2]/env[3]
                if args['gridworld']==1:
                    newenv = gym.make(env_id_1,\
                                  map_width = width, map_height = height,\
                                  items_quantity = {'tree': no_trees, 'rock': no_rocks, 'rubber_tree':no_rubber_tree,'crafting_table': crafting_table, 'fence': no_fence, 'pogo_stick':0},
                                  initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks,'rubber_tree': starting_rubber_trees, 'crafting_table': 0, 'fence': starting_fence, 'pogo_stick':0},\
                                  goal_env = type_of_env, is_final = final_status)
                elif args['gridworld']==2:
                    newenv = gym.make(env_id_2,\
                                  map_width = width, map_height = height,\
                                  items_quantity = {'tree': no_trees, 'rock': no_rocks, 'rubber_tree':no_rubber_tree,'crafting_table': crafting_table, 'oak_tree': no_oak_tree, 'pogo_stick':0},
                                  initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks,'rubber_tree': starting_rubber_trees, 'crafting_table': 0, 'oak_tree': starting_oak_tree, 'pogo_stick':0},\
                                  goal_env = type_of_env, is_final = final_status)
                elif args['gridworld']==3:
                    newenv = gym.make(env_id_3,\
                                  map_width = width, map_height = height,\
                                  items_quantity = {'tree': no_trees, 'rock': no_rocks, 'rubber_tree':no_rubber_tree,'crafting_table': crafting_table, 'pogo_stick':0, 'hard_crafting_table': hard_crafting_table},
                                  initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks,'rubber_tree': starting_rubber_trees, 'crafting_table': 0, 'pogo_stick':0, 'hard_crafting_table':0},\
                                  goal_env = type_of_env, is_final = final_status)
                else:
                    newenv = gym.make(env_id_0,\
                                  map_width = width, map_height = height,\
                                  items_quantity = {'tree': no_trees, 'rock': no_rocks, 'rubber_tree':no_rubber_tree,'crafting_table': crafting_table, 'pogo_stick':0},
                                  initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks,'rubber_tree': starting_rubber_trees, 'crafting_table': 0, 'pogo_stick':0},\
                                  goal_env = type_of_env, is_final = final_status)

                    
                ### generated expaned agent
                ### model_file and weights ????
                ###################################
                new_agent=generate_expanded_agent(env, newenv, agent, parameter_list, 'mid', args['weights'])
                env=newenv
                agent=new_agent
                env.reset()
            if episode > EPISODES:
                agent.save_model(0,0,1,'final')
                break
                
                
    






        
