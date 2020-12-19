'''
Adapted from Jivko Sinapov's implementation of SimpleDQN
https://www.eecs.tufts.edu/~jsinapov/teaching/comp150_RL_Fall2019/
An adaption from Andrej Karpathy's blog
http://karpathy.github.io/2016/05/31/rl/

For questions contact shivam.goel@tufts.edu
'''

import numpy as np
import math
import time
import os
#from chainer import cuda

#import cupy as cp

#backend
#be = "gpu"
#device = 0


be = "cpu"

def ranked_prob(num_actions, ranking, rank_factor=0.2):
	# Action: {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Break', 4: 'Crafting'}
	# ranking array has to be in the same order as the action array
	base_prob = (1 - rank_factor) / num_actions
	denom = num_actions * (num_actions + 1) / 2
	ranked_probs = list(map(lambda r : base_prob + (1/r) * rank_factor, ranking))
	return ranked_probs

class RegularPolicyGradient(object):
	# constructor
	def __init__(self, num_actions, input_size, hidden_layer_size, learning_rate,
				 gamma, decay_rate, greedy_e_epsilon, random_seed):
		# store hyper-params
		self._A = num_actions
		self._D = input_size
		self._H = hidden_layer_size
		self._learning_rate = learning_rate
		self._decay_rate = decay_rate
		self._gamma = gamma
		
		self.explore_aprobs = [1.0 / num_actions] * num_actions

		# hyperparameters for clever exploration
		# explore_type can be 0 (normal) or 1 (clever)
		self.explore_type = 0
		self.clever_episode = 0

		self.rho = None
		self.min_rho = None
		self.max_rho = None
		self.rho_lambda = None
		self.rho_stop = None

		self.clever_lambda = None
		self.clever_stop = None
		self.init_clever_aprobs = None
		self.curr_clever_aprobs = None
		self.block_in_front_offset = None
		self.new_obj_ind = None
		
		# some temp variables
		self._xs,self._hs,self._dlogps,self._drs = [],[],[],[]

		# variables governing exploration
		self._exploration = True # should be set to false when evaluating
		self._explore_eps = greedy_e_epsilon
		
		#create model
		self.init_model(random_seed)
		
		self.log_dir = 'results'
		self.env_id = 'NovelGridworld-v0'
		os.makedirs(self.log_dir, exist_ok = True)

		# successful trajectories
		self._as = [] # list of actions taken this episode
		self.succ_trajectories = []

	def init_model(self,random_seed):
		# create model
		#with cp.cuda.Device(0):
		self._model = {}
		np.random.seed(random_seed)
	   
		# weights from input to hidden layer   
		self._model['W1'] = np.random.randn(self._D,self._H) / np.sqrt(self._D) # "Xavier" initialization
	   
		# weights from hidden to output (action) layer
		self._model['W2'] = np.random.randn(self._H,self._A) / np.sqrt(self._H)
			
		# print("model is: ", self._model)
		# time.sleep(5)		
		self._grad_buffer = { k : np.zeros_like(v) for k,v in self._model.items() } # update buffers that add up gradients over a batch
		self._rmsprop_cache = { k : np.zeros_like(v) for k,v in self._model.items() } # rmsprop memory
	
	# softmax function
	def softmax(self,x):
		probs = np.exp(x - np.max(x, axis=1, keepdims=True))
		probs /= np.sum(probs, axis=1, keepdims=True)
		return probs
		
	def discount_rewards(self,r):
		""" take 1D float array of rewards and compute discounted reward """
		discounted_r = np.zeros_like(r)
		running_add = 0
		for t in reversed(range(0, r.size)):
			running_add = running_add * self._gamma + r[t]
			discounted_r[t] = float(running_add)

		return discounted_r
	
	# feed input to network and get result
	def policy_forward(self,x):
		if(len(x.shape)==1):
			x = x[np.newaxis,...]

		h = x.dot(self._model['W1'])
		
		if np.isnan(np.sum(self._model['W1'])):
			print("W1 sum is nan")
			time.sleep(5)
		if np.isnan(np.sum(self._model['W2'])):
			print("W2 sum is nan")
		
		if np.isnan(np.sum(h)):
			print("nan")
			
			h[np.isnan(h)] = np.random.random_sample()
			h[np.isinf(h)] = np.random.random_sample()
			

		if np.isnan(np.sum(h)):
			print("Still nan!")
		
		
		h[h<0] = 0 # ReLU nonlinearity
		logp = h.dot(self._model['W2'])

		p = self.softmax(logp)
  
		return p, h # return probability of taking actions, and hidden state
		
	def policy_backward(self,eph, epdlogp):
		""" backward pass. (eph is array of intermediate hidden states) """
		dW2 = eph.T.dot(epdlogp)  
		dh = epdlogp.dot(self._model['W2'].T)
		dh[eph <= 0] = 0 # backpro prelu
  
		t = time.time()
  
		if(be == "gpu"):
		  self._dh_gpu = cuda.to_gpu(dh, device=0)
		  self._epx_gpu = cuda.to_gpu(self._epx.T, device=0)
		  self._dW1 = cuda.to_cpu(self._epx_gpu.dot(self._dh_gpu) )
		else:
		  self._dW1 = self._epx.T.dot(dh)

		#print((time.time()-t0)*1000, ' ms, @final bprop')

		return {'W1':self._dW1, 'W2':dW2}
	
	def set_explore_epsilon(self,e):
		self._explore_eps = e

	def set_clever_exploration(self, explore_type, min_rho, max_rho, rho_lambda,
								rho_stop, init_clever_aprobs, clever_lambda,
								clever_stop, block_in_front_offset, new_obj_ind):
		self.explore_type = explore_type
		self.rho = max_rho
		self.max_rho = max_rho
		self.min_rho = min_rho
		self.rho_lambda = rho_lambda
		self.rho_stop = rho_stop
		self.init_clever_aprobs = init_clever_aprobs
		self.curr_clever_aprobs = init_clever_aprobs
		self.clever_lambda = clever_lambda
		self.clever_stop = clever_stop
		self.block_in_front_offset = block_in_front_offset
		self.new_obj_ind = new_obj_ind
	
	# input: current state/observation
	# output: action index
	def process_step(self, x, exploring):

		# feed input through network and get output action distribution and hidden layer
		aprob, h = self.policy_forward(x)
		#print(aprob)
		
		# if exploring
		if exploring == True:
			# greedy-e exploration
			rand_e = np.random.uniform()
		
			if self.explore_type == 1: # clever exploration based on ranking
				# check if in front of new object
				if self.get_block_in_front(x) == self.new_obj_ind:
					# check if random value less than new_object epsilon (rho)
					if rand_e < self.rho: 
						aprob[0] = self.explore_aprobs
						# use ranking-based probs if not past clever episode limit
						if self.clever_episode < self.clever_stop:
							aprob[0] = list(self.curr_clever_aprobs)
				else: # use epsilon if not in front of new object
					if rand_e < self._explore_eps:
						aprob[0] = self.explore_aprobs
			else: # doing regular exploration (explore_type == 0)
				if rand_e < self._explore_eps:
					aprob[0] = self.explore_aprobs

		if np.isnan(np.sum(aprob)):
			print(aprob)
			aprob[0] = [ 1.0/len(aprob[0]) for i in range(len(aprob[0]))]
			print(aprob)
			#input()
		
		aprob_cum = np.cumsum(aprob)
		u = np.random.uniform()
		a = np.where(u <= aprob_cum)[0][0]	

		# record various intermediates (needed later for backprop)
		t = time.time()
		self._xs.append(x) # observation
		self._hs.append(h)

		#softmax loss gradient
		dlogsoftmax = aprob.copy()
		dlogsoftmax[0,a] -= 1 #-discounted reward 
		self._dlogps.append(dlogsoftmax)
		
		t  = time.time()

		self._as.append(a)

		return a

	def get_block_in_front(self, x):
		block_in_front_v = x[self.block_in_front_offset :]
		# check if in front of some block
		id_block_in_front = np.where(block_in_front_v == 1)
		if len(id_block_in_front[0]):
			return id_block_in_front[0][0]
		else:
			return -1

	# after process_step, this function needs to be called to set the reward
	def give_reward(self,reward):
		
		# store the reward in the list of rewards
		self._drs.append(reward)

	# reset to be used when evaluating
	def reset(self):
		self._xs,self._hs,self._dlogps,self._drs = [],[],[],[] # reset 
		self._grad_buffer = { k : np.zeros_like(v) for k,v in self._model.items() } # update buffers that add up gradients over a batch
		self._rmsprop_cache = { k : np.zeros_like(v) for k,v in self._model.items() } # rmsprop memory
		
	# this function should be called when an episode (i.e., a game) has finished
	def finish_episode(self, episode_done):
		# stack together all inputs, hidden states, action gradients, and rewards for this episode
		
		# this needs to be stored to be used by policy_backward
		# self._xs is a list of vectors of size input dim and the number of vectors is equal to the number of time steps in the episode
		self._epx = np.vstack(self._xs)
		
		#for i in range(0,len(self._hs)):
		#	print(self._hs[i])
		
		# len(self._hs) = # time steps
		# stores hidden state activations
		eph = np.vstack(self._hs)
		
		#for i in range(0,len(self._dlogps)):
		#	print(self._dlogps[i])
		
		# self._dlogps stores a history of the probabilities over actions selected by the agent
		epdlogp = np.vstack(self._dlogps)
		
		# self._drs is the history of rewards
		#for i in range(0,len(self._drs)):
		#	print(self._drs[i])
		epr = np.vstack(self._drs)
		
		# compute the discounted reward backwards through time
		discounted_epr = (self.discount_rewards(epr))

		if episode_done: # if episode ended due to reaching goal
			self.succ_trajectories.append((np.vstack(self._xs), self._as, 
										   discounted_epr))

		self._xs, self._hs, self._as, self._dlogps, self._drs = [],[],[],[],[] # reset array memory

		#for i in range(0,len(discounted_epr)):
		#	print(str(discounted_epr[i]) + "\t"+str(epr[i]))
		
		# #print(discounted_epr)
		# discounted_epr_mean = np.mean(discounted_epr)
		# #print(discounted_epr_mean)
		
		# # standardize the rewards to be unit normal (helps control the gradient estimator variance)
		
		# #discounted_epr -= np.mean(discounted_epr)
		# discounted_epr = np.subtract(discounted_epr,discounted_epr_mean)
		
		# discounted_epr /= np.std(discounted_epr)+0.01
		
		epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
		
		start_time = time.time()
		grad = self.policy_backward(eph, epdlogp)
		#print("--- %s seconds for policy backward ---" % (time.time() - start_time))
		
		for k in self._model: self._grad_buffer[k] += grad[k] # accumulate grad over batch

		# decay curr_clever_aprobs towrd explore_aprobs and decay rho towards min_rho
		# if doing cleveration exploration and not at respective stop episodes
		# TODO - ensure this works
		if self.explore_type == 1:
			if self.clever_episode < self.clever_stop:
				self.curr_clever_aprobs = np.array(self.explore_aprobs) + \
					(self.init_clever_aprobs - np.array(self.explore_aprobs)) * \
						math.exp(-self.clever_lambda * self.clever_episode)	
			if self.clever_episode < self.rho_stop:
				self.rho = self.min_rho + \
					(self.max_rho - self.min_rho) * \
						math.exp(-self.rho_lambda* self.clever_episode)
			self.clever_episode += 1

	# upon addition of new items, called to expand network with given number of new
	# input nodes, and connections to hidden layer initialized with random weights
	def expand_random_weights(self, num_new_inputs=None):
		# determine number of nodes to add
		if num_new_inputs is None:
			num_new_inputs = self._D - int(self._model['W1'].shape[0])
		elif self._D != int(self._model['W1'].shape[0]) + num_new_inputs:
			print("[expand_random_weights] Warning: num_new_inputs chosen such that"
				  "model input layer size will not equal self._D after expansion")

		if num_new_inputs != 0:
			to_append = np.random.randn(num_new_inputs, self._H)/np.sqrt(num_new_inputs)
			self._model['W1'] = np.vstack((self._model['W1'], to_append))

	def expand_copy_weights(self, copy_object_indices):
		# TODO - consider adding some randomization to the weights
		for ind in copy_object_indices:
			self._model['W1'] = np.vstack((self._model['W1'], self._model['W1'][ind]))

	# called to update model parameters, generally every N episodes/games for some N
	def update_parameters(self):
		for k,v in self._model.items():
			g = self._grad_buffer[k] # gradient
			self._rmsprop_cache[k] = self._decay_rate * self._rmsprop_cache[k] + (1 - self._decay_rate) * g**2
			self._model[k] -= self._learning_rate * g / (np.sqrt(self._rmsprop_cache[k]) + 1e-5)
			self._grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

## TODO: adapt to old style
	def save_model(self, curriculum_no, beam_no, env_no, ep_number):
		if ep_number =='final':
			experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no) + '_final'

		experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no) +'_ep' + str(ep_number)
		path_to_save = self.log_dir + os.sep + self.env_id + experiment_file_name + '.npz'
		np.savez(path_to_save, layer1 = self._model['W1'], layer2 = self._model['W2'])
		print("saved to: ", path_to_save)

## TODO: adapt to old style
	def load_model(self, curriculum_no, beam_no, env_no, ep_number): 

		experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no) + '_ep'+str(ep_number)
		path_to_load = self.log_dir + os.sep + self.env_id + experiment_file_name + '.npz'
		data = np.load(path_to_load)
		self._model['W1'] = data['layer1']
		self._model['W2'] = data['layer2']
		print ("loaded model from {}".format(path_to_load))

	def load_model_from_dict(self, model_dict):
		self._model['W1'] = model_dict['W1']
		self._model['W2'] = model_dict['W2']
		print ("loaded model from model dictionary")