
'''
This code evaluates the learned models
on novelgridworlds environment task (novel_gridworld_v0_env.py)
using Regular Policy Gradient algorithm (RegularPolicyGradient.py)
Saves the csv(s) for plotting in results/data/<test_results.csv>

run this to evaluate 
$ python evaluate.py -E <number_of _episodes_to_test> -M <model_episode_number> -R <Render(True/False)>

For questions contact shivam.goel@tufts.edu

'''

import os
import csv

import gym
import numpy as np
import gym_novel_gridworlds

from RegularPolicyGradient import RegularPolicyGradient
import matplotlib.pyplot as plt
import argparse



def save_results (data, tag):
    # add the functionality of 
    # searching and making dir based on the time and 
    #date of run and also on the bases of the algo run
    os.makedirs("results" + os.sep + "data", exist_ok=True)

    if tag == 'test_results':
        db_file_name = "results/data/test_results.csv"
        with open(db_file_name, 'a') as f: # append to the file created
            writer = csv.writer(f)
            writer.writerow(data)

def evaluate (dqn_agent, env, evaluate = True, demo = False, render = False):
    # exploration
    explore = True if evaluate==False else False
    obs = env.get_observation()
    R = 0
    r = 0
    count = 0
    D = False
        
    #print ("Observation space dims = {}".format(obs.shape))
    for step in range(EPISODE_SIZE):

        a = dqn_agent.process_step(obs, explore)
        #print ("Action INDEX = {}".format(a))
        #print ("ACTION = {}".format(move_actions[a]))
        obs2, reward, done, info = env.step(a)
        # print ("Observation = {}".format(obs2))
        # print ("Reward = {}".format(r))
        # print ("Done = {}".format(done)) 

        if render == True:
            env.render()

        agent.give_reward(reward) #give reward
        R += reward
        #print(count) 
        # agent.compute_reward(r)

        if done:
            D = True
            count = step
            if evaluate:
                dqn_agent.reset()
            else:
                dqn_agent.finish_episode()
            break

        obs = obs2
        count += 1

    if evaluate:
        return[R, count, D]

if __name__ == "__main__":
    # get all the args
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--episodes", default=100, help="Number of episodes you need to evaluate the model", type = int)
    ap.add_argument("-M", "--model", default= 'final', help="Episode number of the model you want to load")
    ap.add_argument("-R","--render", default = False, help="Want to render or not(1 or 0)")
    ap.add_argument("-print_output", default="", help="print stuff")
    args = vars(ap.parse_args())

    # agent variables
    actionCnt = 5
    D = 46 #8 beams x 5 items lidar + 6 inventory items
    NUM_HIDDEN = 20
    GAMMA = 0.95
    LEARNING_RATE = 1e-3
    DECAY_RATE = 0.99
    MAX_EPSILON = 0.1
    random_seed = 1
    random_seed = 1

    # environment variables
    env_id = 'NovelGridworld-v0'
    final_status = True # If True, reward shaping present in the task.
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

    # parameters
    EPISODE_SIZE = 150

    no_tests = int(args['episodes']) # number of episodes for evaluations

    # get the agent
    agent = RegularPolicyGradient(actionCnt,D,NUM_HIDDEN,\
                     LEARNING_RATE,GAMMA,DECAY_RATE,\
                     MAX_EPSILON,random_seed)
    agent.set_explore_epsilon(MAX_EPSILON)
    # add the episode number based on the model we want to load
    agent.load_model(curriculum_no = 0, beam_no = 0, env_no = 1, ep_number=args['model'])

   # get the environment
    env = gym.make(env_id,\
                  map_width = width, map_height = height,\
                  items_quantity = {'tree': no_trees, 'rock': no_rocks, 'rubber_tree':no_rubber_tree,'crafting_table': crafting_table, 'pogo_stick':0},\
                  initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks,'rubber_tree': starting_rubber_trees, 'crafting_table': 0, 'pogo_stick':0},\
                  goal_env = type_of_env, is_final = final_status)
    # run evaluations for number of episodes
    ctr = 0
    for i in range(no_tests):
        env.reset()
        result = evaluate(agent, env, evaluate=True, render=args['render'])
        print("Episode--> {} cum_reward--> {} steps-->> {} Done = {}".format(i+1,result[0],result[1], result[2]))
        # write the test csv
        data = [result[0], result[1]]
        # check if the task is successful and count
        if result[2] == True:
            ctr+=1
        save_results(data, tag = 'test_results')     
    print ("Agent FINISHED the TASK {} out of {} trials".format(ctr, no_tests))