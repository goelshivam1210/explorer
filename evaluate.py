
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
from params_baseline import *
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
    EPISODE_SIZE = 150   
    #print ("Observation space dims = {}".format(obs.shape))
    for step in range(EPISODE_SIZE):

        a = dqn_agent.process_step(obs, explore)
        #print ("Action INDEX = {}".format(a))
        #print ("ACTION = {}".format(move_actions[a]))
        obs2, reward, done, info = env.step(a)
        # print ("Observation = {}".format(obs2))
        # print ("Reward = {}".format(r))
        # print ("Done = {}".format(done)) 
        if render:
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

    ap.add_argument("-G", "--gridworld", default = 0, help="New (novel) env to be loaded (0, 1, 2, or 3)", type=int)
    ap.add_argument("-S", "--sim_weights", default=False, help="Use weights from similar object for expanding network", type=bool)
    ap.add_argument("-Q", "--clever", default=False, help="Use ranking-augmented probabilities clever exploration after adaptation", type=bool)

    args = vars(ap.parse_args())

    # load the learning agent
    
    agent_params = {
        "num_actions": actionCnt, 
        "input_size": D, 
        "hidden_layer_size": NUM_HIDDEN,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "decay_rate": DECAY_RATE,
        "greedy_e_epsilon": 0.01,
        "random_seed": random_seed
    }
    agent = RegularPolicyGradient(**agent_params)
    agent.set_explore_epsilon(0.01) # set the epsilon to be 0.01: No exploration
    
    # load the model from correct file
    if (args['sim_weights'] == True and args['clever'] != True):
        agent.load_model(0,0,args['gridworld']+20000, args['model'])
    elif (args['clever'] == True and args['sim_weights'] == True):
            agent.load_model(0,0,args['gridworld']+50000, args['model'])
    else:
        agent.load_model(0,0,args['gridworld']+20, args['model'])
    
    # load the correct environment
    if args['gridworld']==1:
        env = gym.make(env_id_1, map_width=width, map_height=height, goal_env=type_of_env, is_final=final_status)
        print(f"Starting environment {env_id_1}")
    elif args['gridworld']==2:
        env = gym.make(env_id_2, map_width=width, map_height=height, goal_env=type_of_env, is_final=final_status)
        print(f"Starting environment {env_id_2}")
    elif args['gridworld']==3:
        env = gym.make(env_id_3, map_width=width, map_height=height, goal_env=type_of_env, is_final=final_status)
        print(f"Starting environment {env_id_3}")
    else:
        env = gym.make(env_id_0, map_width=width, map_height=height, goal_env=type_of_env, is_final=final_status)
        print(f"Starting with environment {env_id_0}")

    no_tests = int(args['episodes']) # number of episodes for evaluations
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