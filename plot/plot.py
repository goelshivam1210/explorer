
'''
This code takes in the results file, 
npy or csv and other formats and 
generates plots as we learn.
run this while training 
$ python plot.py <Window_Size>

For questions contact shivam.goel@tufts.edu

'''


import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set()
print (sys.argv[1])
plt.style.use('seaborn-whitegrid')
window_size = int(sys.argv[1])
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
            #a = line.split(',')
            reward = float(line[1])
            epsilon = float(line[2])
            R.append(reward)
            E.append(epsilon)
            #print(line)

    #print (R)
    R_sum = []
    E_avg = []
    for i in range (len(R) - window_size):
        a = R[i:i+window_size]
        e = E[i:i+window_size]
        rolling_sum = np.mean(a)
        epsilon_sum = np.mean(e)
        R_sum.append(rolling_sum)
        E_avg.append(epsilon_sum)

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

    plt.plot(R_sum)
    # plt.plot(E_avg)
    plt.xlabel("Number of episodes")
    # plt.xlabel("Number of time steps")
    plt.ylabel("Average cumulative reward per episode")
    plt.grid(True)
    #plt.title("TO-SARSA-λ learning curve for polycraft:pogo_task(incremental)(Window-Size ="+str(window_size)+")")
    plt.title("Learning curve:default_env:pogostick (WS ="+str(window_size)+")")
    plt.tight_layout()
    # update every 15 minutes (900*60)
    plt.pause(900)
    plt.clf()
    sns.set()
    #plt.pause()
# plot_colortable(mcolors.TABLEAU_COLORS, "Tableau Palette",
#                 sort_colors=False, emptycols=2)
#plt.savefig("learning_curve_running" +str(len(R_sum))+str(window_size)+".png")
plt.show()


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