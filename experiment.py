
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
from explorer import generate_adaptive_agent
from params_baseline import *
import matplotlib.pyplot as plt

def save_results (data, tag, tag2, tag3):
    os.makedirs("plot" + os.sep + "data", exist_ok=True)
    if tag == 'train_results':
        db_file_name = "plot/data/train_results_" + str(tag2)+ str(tag3)+".csv"
        with open(db_file_name, 'a') as f: # append to the file created
            writer = csv.writer(f)
            writer.writerow(data)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-C", "--continue", default=False, help="Want to continue training or start from scratch", type=bool)
    ap.add_argument("-M", "--model", default= 'final', help="Episode number of the model you want to load")
    ap.add_argument("-E", "--episode", default= 0, help="Episode number to resume from (used for saving results)", type=int)
    ap.add_argument("-R","--render", default = False, help="Want to render or not (False or True)", type = bool)
    ap.add_argument("-P", "--print_every", default= 100, help="Number of epsiodes you want to print the results", type=int)
    ap.add_argument("-N", "--num_model", default= 4, help="Number of models want to save before final trained model. Used for evaluations", type=int)
    # --- Novelty adaptation specific arguments --- #
    ap.add_argument("-G", "--gridworld", default= 0, help="New (novel) env to be loaded (0, 1, 2, or 3)", type=int)
    ap.add_argument("-T", "--trials_novelty", default=700, help="Number of trials (episodes) to run before switching envs", type=int)
    ap.add_argument("-S", "--sim_weights", default=False, help="Use weights from similar object for expanding network", type=bool)

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

    # NOTE - use this to load a pre-trained model before doing experiment
    if args['continue'] == True:
        print ("LOADING model ....")
        agent.load_model(curriculum_no = 0, beam_no = 0, env_no = 1, ep_number=args['model'])

    # get the environment
    env = gym.make(env_id_0, map_width=width, map_height=height, goal_env=type_of_env, is_final=final_status)
    print(f"Starting environment {env_id_0}")
    
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
            save_results(data, tag = 'train_results', tag2 = args['gridworld'], tag3 = args['sim_weights'])

            env.reset()
            reward_sum = 0
            # save only 4 models
            if episode% (EPISODES/args['num_model']) == 0:
                if args['sim_weights'] == True:
                    agent.save_model(0,0,args['gridworld']+20000,episode)
                else:
                    agent.save_model(0,0,args['gridworld']+20,episode)
    
            # change to a new_env after 700 trials (episodes)
            # TODO - turn into an argument (or change back to 700)
            if episode == args['trials_novelty']:
                ######## change the file name here
                if args['sim_weights'] == True:
                    agent.save_model(0,0,args['gridworld']+20000,'mid')
                else:
                    agent.save_model(0,0,args['gridworld']+20,'mid') # Harcoded for now

                if args['gridworld']==1:
                    newenv = gym.make(env_id_1, map_width=width, map_height=height, goal_env=type_of_env, is_final=final_status)
                    print(f"Starting environment {env_id_1}")
                elif args['gridworld']==2:
                    newenv = gym.make(env_id_2, map_width=width, map_height=height, goal_env=type_of_env, is_final=final_status)
                    print(f"Starting environment {env_id_2}")
                elif args['gridworld']==3:
                    newenv = gym.make(env_id_3, map_width=width, map_height=height, goal_env=type_of_env, is_final=final_status)
                    print(f"Starting environment {env_id_3}")
                else:
                    newenv = gym.make(env_id_0, map_width=width, map_height=height, goal_env=type_of_env, is_final=final_status)
                    print(f"Continuing with environment {env_id_0}")

                # generate the expanded agent
                new_agent=generate_adaptive_agent(env, newenv, agent, parameter_list, args['sim_weights'])
                env=newenv
                agent=new_agent
                env.reset()
            if episode > EPISODES:
                if args['sim_weights'] == True:
                    agent.save_model(0,0,args['gridworld']+20000,episode)
                else:
                    agent.save_model(0,0,args['gridworld']+20,'final')
                break
                
                
    






        
