'''
This code takes in the results file, 
npy or csv and other formats and 
generates plots as we learn.
run this while training 

$ python plot.py -W <window_size> -P <pause_time(s)>

For questions contact shivam.goel@tufts.edu
'''

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set()
plt.style.use('seaborn-whitegrid')

ap = argparse.ArgumentParser()
ap.add_argument("-W", "--window", default=50, help="Window Size for smoothing in plotting", type=int)
ap.add_argument("-P", "--pause", default=900, help="pause time (s) for the plot to wait before updating", type=int)
ap.add_argument("-print_output", default="", help="print stuff")
args = vars(ap.parse_args())

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# there are some duplicate and bad entries after 
# 130K episodes so use this to remove them
def clean_data(arr):
    # this is HACK
    for j in range(3):
        for i in range(int(arr.shape[0])):
            if (i > 100000 and i < 160000):
                if int(arr[i][0]) < 2000:
                    arr = np.delete(arr, (i), axis=0)

    arr = np.delete(arr, (0), axis=1)
    arr = arr.reshape(-1)
    return arr

def parser(filename):

    R = []
    # E = []
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        for line in reader:
            if line:
                ep_num = float(line[0])
                reward = float(line[1])
                # epsilon = float(line[2])
                R.append((ep_num, reward))

                # E.append(epsilon)
                #print(line)
    R = np.array(R)
    # print (R.shape, R[0], R[1])
    # E = np.array(E)
    R = clean_data(R)
    R_sum = moving_average(R, args['window'])
    # E_avg = moving_average(E, args['window'])
    return R_sum

if __name__ == '__main__':
# domain v1
    exp_b_v1_file = 'data/train_results_1False.csv'
    exp_c_v1_file = 'data/train_results_1True.csv'
    exp_d_v1_file = 'data/train_results_1Falseclever.csv'
    exp_e_v1_file = 'data/train_results_1Trueclever.csv'

# domain v2
    exp_b_v2_file = 'data/train_results_2False.csv'
    exp_c_v2_file = 'data/train_results_2True.csv'
    exp_d_v2_file = 'data/train_results_2Falseclever.csv'
    exp_e_v2_file = 'data/train_results_2Trueclever.csv'

# domain v3
    exp_b_v3_file = 'data/train_results_3False.csv'
    exp_c_v3_file = 'data/train_results_3True.csv'
    exp_d_v3_file = 'data/train_results_3Falseclever.csv'
    exp_e_v3_file = 'data/train_results_3Trueclever.csv'

# get all the data
    exp_b_v1 = parser(exp_b_v1_file)
    exp_c_v1 = parser(exp_c_v1_file)
    exp_d_v1 = parser(exp_d_v1_file)
    exp_e_v1 = parser(exp_e_v1_file)

    exp_b_v2 = parser(exp_b_v2_file)
    exp_c_v2 = parser(exp_c_v2_file)
    exp_d_v2 = parser(exp_d_v2_file)
    exp_e_v2 = parser(exp_e_v2_file)

    exp_b_v3 = parser(exp_b_v3_file)
    exp_c_v3 = parser(exp_c_v3_file)
    exp_d_v3 = parser(exp_d_v3_file)
    exp_e_v3 = parser(exp_e_v3_file)


    '''
    Nuisance Novelty Domain
    '''
    plt.plot(exp_b_v1, label = 'Random weights')
    plt.plot(exp_c_v1, label = 'Similar weights')
    plt.plot(exp_d_v1, label = 'Clever-Explorer-Random')
    plt.plot(exp_e_v1, label = 'Clever-Explorer-Similar')

    plt.xlabel("Number of episodes")
    plt.ylabel("Average cumulative reward per episode")
    plt.grid(True)
    plt.legend(loc = 7)
    # plt.yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
    plt.xlim(xmin=0.0, xmax=310000)
    plt.tight_layout()
    plt.savefig('nuisance_novelty_WS='+str(args['window'])+'.png', dpi = 800)
    plt.close()
    
    '''
    Shortcut Novelty Domain
    '''
    plt.plot(exp_b_v2, label = 'Random weights')
    plt.plot(exp_c_v2, label = 'Similar weights')
    plt.plot(exp_d_v2, label = 'Clever-Explorer-Random')
    plt.plot(exp_e_v2, label = 'Clever-Explorer-Similar')

    plt.xlabel("Number of episodes")
    plt.ylabel("Average cumulative reward per episode")
    plt.grid(True)
    plt.legend(loc = 7)
    # plt.yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
    plt.xlim(xmin=0.0, xmax=310000)
    plt.tight_layout()
    plt.savefig('shortcut_novelty_WS='+str(args['window'])+'.png', dpi = 800)
    plt.close()

    '''
    Catastrophic Novelty Domain
    '''
    plt.plot(exp_b_v3, label = 'Random weights')
    plt.plot(exp_c_v3, label = 'Similar weights')
    plt.plot(exp_d_v3, label = 'Clever-Explorer-Random')
    plt.plot(exp_e_v3, label = 'Clever-Explorer-Similar')

    plt.xlabel("Number of episodes")
    plt.ylabel("Average cumulative reward per episode")
    plt.grid(True)
    plt.legend(loc = 7)
    # plt.yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
    plt.xlim(xmin=0.0, xmax=310000)
    plt.tight_layout()
    plt.savefig('catastrophic_novelty_WS='+str(args['window'])+'.png', dpi = 800)

    plt.close()

'''
# uncomment code block for live plotting
R = []
while True:
    R = []
    E = []
    #xfile_name = "../results/data"
    file_name = "data/train_results.csv"
    with open(file_name, mode='r') as infile:
        reader = csv.reader(infile)
        #reader = infile.readlines()[1:] # read each line as string and skip first line
        for line in reader:
            if line:
                #a = line.split(',')
                reward = float(line[1])
                epsilon = float(line[2])
                R.append(reward)
                E.append(epsilon)
                #print(line)

    #print (R)
    R = np.array(R)
    E = np.array(E)
    R_sum = moving_average(R, args['window'])
    E_avg = moving_average(E, args['window'])

    # R_sum = []
    # E_avg = []
    # for i in range (len(R) - int(args['window'])):
    #     a = R[i:i+int(args['window'])]
    #     e = E[i:i+int(args['window'])]
    #     rolling_sum = np.mean(a)
    #     epsilon_sum = np.mean(e)
    #     R_sum.append(rolling_sum)
    #     E_avg.append(epsilon_sum)

    #plot

# x_axis = np.arange(len(R_sum)*50)
# fig, ax = plt.subplots()
# for i in range(len())

    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('Episodes')
    # ax1.set_ylabel('Average cumulative reward per episode')
    # ax1.plot(R_sum) #color= 'pink')
    # ax1.tick_params(axis= 'y') #labelcolor = 'pink')
    # # use the twin plot using same x-axis
    # ax2 = ax1.twinx()
    # # color = 'grey'
    # ax2.set_ylabel('Epsilon')
    # ax2.plot(E_avg, color = 'indianred')
    # ax2.tick_params(axis='y') #labelcolor=color)
    # fig.tight_layout()

    plt.plot(R_sum, label = 'Simple-Policy-Gradient')
    # plt.plot(E_avg)
    plt.xlabel("Number of episodes")
    # plt.xlabel("Number of time steps")
    plt.ylabel("Average cumulative reward per episode")
    plt.grid(True)
    plt.legend(loc = 7)
    #plt.title("TO-SARSA-λ learning curve for polycraft:pogo_task(incremental)(Window-Size ="+str(int(args['window']))+")")
    plt.title("Learning performance:default_env:easy_pogostick (WS = "+str(int(args['window']))+")")
    plt.tight_layout()
    # update every 15 minutes (900*60)
    plt.pause(int(args['pause']))
    plt.clf()
    sns.set()
    #plt.pause()
# plot_colortable(mcolors.TABLEAU_COLORS, "Tableau Palette",
#                 sort_colors=False, emptycols=2)
#plt.savefig("learning_curve_running" +str(len(R_sum))+str(window_size)+".png")
plt.show()
'''


#fig, ax = plt.subplots()

#for i in range(len(data)):
#    ax.cla()
#    ax.imshow(data[i])
#    ax.set_title("frame {}".format(i))
    # Note that using time.sleep does *not* work here!
#    plt.pause(0.1)


# import numpy as np
# import matplotlib.pyplot as plt

# plt.axis([0, 10, 0, 1])

# for i in range(10):
#     y = np.random.random()
#     plt.scatter(i, y)
#     plt.pause(8*60)

# plt.show()

# plot_colortable(mcolors.TABLEAU_COLORS, "Tableau Palette",
#                 sort_colors=False, emptycols=2)