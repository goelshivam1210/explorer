
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

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-C", "--continue", default=False, help="Want to continue training or start from scratch", type=bool)
    ap.add_argument("-M", "--model", default= 'final', help="Episode number of the model you want to load")
    ap.add_argument("-E", "--episode", default= 0, help="Episode number from to resume(used for saving results)", type=int)
    ap.add_argument("-R","--render", default = False, help="Want to render or not(False or True)", type = bool)
    ap.add_argument("-P", "--print_every", default= 100, help="Number of epsiodes you want to print the results", type=int)
    ap.add_argument("-N", "--num_model", default= 4, help="Number of models want to save before final trained model. Used for evaluations", type=int)

    ap.add_argument("-print_output", default="", help="print stuff")
    args = vars(ap.parse_args())

    # load the learning agent
    agent_params = {
        "num_actions": actionCnt, 
        "input_size": D, 
        "hidden_layer_size": NUM_HIDDEN,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "decay_rate": DECAY_RATE,
        "greedy_e_epsilon": MAX_EPSILON,
        "random_seed": random_seed,
    }

    # NOTE - params for running on V1Env, V2Env, & v3Env without adaptation first
    # agent_params = {
    #     "num_actions": actionCnt, 
    #     "input_size": 62, 
    #     "hidden_layer_size": NUM_HIDDEN,
    #     "learning_rate": LEARNING_RATE,
    #     "gamma": GAMMA,
    #     "decay_rate": DECAY_RATE,
    #     "greedy_e_epsilon": MAX_EPSILON,
    #     "random_seed": random_seed
    # }

    agent = RegularPolicyGradient(**agent_params)

    # agent = RegularPolicyGradient(actionCnt,D,NUM_HIDDEN,\
    #                  LEARNING_RATE,GAMMA,DECAY_RATE,\
    #                  MAX_EPSILON,random_seed)

    if args['continue'] == True:
        print ("LOADING model ....")
        agent.load_model(curriculum_no = 0, beam_no = 0, env_no = 1, ep_number=args['model'])
    
    # NOTE - this is for testing other environments
    # env_id = 'NovelGridworld-v3'
    # env = gym.make(env_id, map_width = width, map_height = height, 
                #    goal_env = type_of_env, is_final = final_status)

    # get the environment
    env = gym.make(env_id,\
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
    
            # quit after some number of episodes
            if episode > EPISODES:
                agent.save_model(0,0,1,'final') # Harcoded for now
                break