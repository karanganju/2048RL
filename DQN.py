import tensorflow as tf
import numpy as np
from env import Env,Agent
from keras.models import Sequential
from keras.layers import Dense
import random
import time
import sys, getopt

# Hyperparams
read_model = False
discount = 0.9
max_epsilon = 1
min_epsilon = 0.1
replay_size = 2048
replay_iters = 128
bsize = 8
save_stops = 50
runs = 0
runs_to_min_epsilon = 100

class Replays(object):

	def __init__(self, size, ssize):
		self.cap = size
		self.sample_size = ssize
		self.iter = -1
		self.filled = False
		self.s = np.zeros([self.cap, 16], dtype=int)
		self.a = np.zeros([self.cap, 1], dtype=int)
		self.r = np.zeros([self.cap, 1], dtype=float)
		self.s_new = np.zeros([self.cap, 16], dtype=int)

	def sample(self):
		ret_s = np.zeros([self.sample_size, 16])
		ret_a = np.zeros([self.sample_size])
		ret_r = np.zeros([self.sample_size])
		ret_s_new = np.zeros([self.sample_size, 16])

		for i in xrange(self.sample_size):
			r = 0
			if (self.filled):
				r = random.randrange(self.cap)
			else:
				r = random.randrange(self.iter+1)

			for j in xrange(16):
				ret_s[i][j] = self.s[r][j]
			for j in xrange(16):
				ret_s_new[i][j] = self.s_new[r][j]
			ret_r[i] = self.r[r]
			ret_a[i] = self.a[r]

		return ret_s, ret_a, ret_r, ret_s_new

	# Much param naming such wow!
	def add_instance(self, arr, x, y, arr2):
		self.iter = self.iter+1
		if (self.iter == self.cap):
			self.filled = True
			self.iter = 0
		for i in xrange(16):
			self.s[self.iter][i] = arr[i]
		for i in xrange(16):
			self.s_new[self.iter][i] = arr[i]
		# Take s,a instead of R+gamma*Q ?
		self.r[self.iter] = y
		self.a[self.iter] = x

class DQN(object):

	def __init__(self, replay):
		
		self.replays = replay

		if (read_model):
			json_file = open('model.json', 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			self.model = model_from_json(loaded_model_json)
			# load weights into new model
			self.model.load_weights("model.h5")
		else:
			self.model = Sequential()
			self.model.add(Dense(16, input_dim=16, init='uniform', activation='relu'))
			self.model.add(Dense(9, init='uniform', activation='relu'))
			self.model.add(Dense(9, init='uniform', activation='relu'))
			self.model.add(Dense(9, init='uniform', activation='relu'))
			self.model.add(Dense(4, init='uniform'))

			self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

	# I'm not sure this works! \-_-/
	# Change of architecture -> Have to look at this. For now it is bs.
	def run_through_replay(self):
		if (replays.filled or replays.iter >= 512):
			s,a,r,s_new = replays.sample()
			
			for replay_iter in xrange(replay_iters):
				sim_state = s_new[replay_iter][:].reshape(16)
				r[replay_iter] += discount * Q_max(sim_state)

			self.model.fit(s, r, nb_epoch=1, batch_size=bsize, verbose=0)

	def select_epsilon_greedy(self, state):
		act_taken = self.select_greedy(state)
		epsilon = max(((min_epsilon - max_epsilon)*runs)/runs_to_min_epsilon + max_epsilon, min_epsilon)

		if (random.random() < epsilon):
			act_taken = random.randrange(4)

		return act_taken

	def select_greedy(self, state):
		Q_vals = self.model.predict(state.reshape([1, 16]))
		return np.argmax(Q_vals)

	def Q_max(self, state):
		Q_vals = self.model.predict(state.reshape([1, 16]))
		return np.max(Q_vals)

	def save_model(self):
		model_json = self.model.to_json()
		with open("model.json", "w") as json_file:
		    json_file.write(model_json)
		self.model.save_weights("model.h5")

if __name__ == '__main__':

	opts, args = getopt.getopt(sys.argv,"i",[])
	for opt, arg in opts:
		if opt == '-i':
			read_model = True

	replays = Replays(replay_size, replay_iters)
	dqn = DQN(replays)

	while(1):

		game = Env(True)
		agent = Agent(game)

		init_state = agent.get_array()
		# Epsilon greedy here.
		init_act = dqn.select_epsilon_greedy(init_state)

		runs += 1
		iters = 0

		# Run through environment
		while(1):

			# Get reward and next state and max - action
			game_state, reward = agent.take_step(init_act)
			# time.sleep(3)
			iters += 1

			# Game over
			if (game_state == -1 or game_state == 1):
				print reward
				print '{0:6d} {1:5d} {2:5d}'.format(game.score, game.max, iters)
				break

			next_state = agent.get_array()
			
			# Add data to mini-batch
			replays.add_instance(init_state, init_act, reward, next_state) 

			# Run through replay
			dqn.run_through_replay()

			# Switch init_* with epsilon greedy
			init_state = next_state[:16]
			init_act = dqn.select_epsilon_greedy(init_state)
			
		if (runs % save_stops == 0):
			dqn.save_model()