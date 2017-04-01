import tensorflow as tf
import numpy as np
from env import Env,Agent
from keras.models import Sequential
from keras.layers import Dense
import random
import time
import sys, getopt

class Replays(object):

	def __init__(self, size, ssize):
		self.cap = size
		self.sample_size = ssize
		self.iter = -1
		self.filled = False
		self.sa = np.zeros([self.cap, 17], dtype=int)
		self.r = np.zeros([self.cap, 1], dtype=float)
		self.s_new = np.zeros([self.cap, 16], dtype=int)

	def sample(self):
		ret_sa = np.zeros([self.sample_size, 17])
		ret_r = np.zeros([self.sample_size])
		ret_s_new = np.zeros([self.sample_size, 16])

		for i in xrange(self.sample_size):
			r = 0
			if (self.filled):
				r = random.randrange(self.cap)
			else:
				r = random.randrange(self.iter+1)

			for j in xrange(17):
				ret_sa[i][j] = self.sa[r][j]
			for j in xrange(16):
				ret_s_new[i][j] = self.s_new[r][j]
			ret_r[i] = self.r[r]

		return ret_sa, ret_r, ret_s_new

	def add_instance(self, arr, y, arr2):
		self.iter = self.iter+1
		if (self.iter == self.cap):
			self.filled = True
			self.iter = 0
		for i in xrange(17):
			self.sa[self.iter][i] = arr[i]
		for i in xrange(16):
			self.s_new[self.iter][i] = arr[i]
		# Take s,a instead of R+gamma*Q ?
		self.r[self.iter] = y


if __name__ == '__main__':

	read_model = False

	opts, args = getopt.getopt(sys.argv,"i",[])
	for opt, arg in opts:
		if opt == '-i':
			read_model = True

	# Hyperparams
	discount = 0.9
	max_epsilon = 1
	min_epsilon = 0.1
	replay_size = 2048
	replay_iters = 128
	bsize = 8
	save_stops = 50
	runs = 0
	runs_to_min_epsilon = 100

	# Define Network
	if (read_model):
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		# load weights into new model
		model.load_weights("model.h5")
	else:
		model = Sequential()
		model.add(Dense(17, input_dim=17, init='uniform', activation='relu'))
		model.add(Dense(9, init='uniform', activation='relu'))
		model.add(Dense(9, init='uniform', activation='relu'))
		model.add(Dense(9, init='uniform', activation='relu'))
		model.add(Dense(4, init='uniform'))

		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
	
	replays = Replays(replay_size, replay_iters)

	while(1):

		game = Env(False)
		agent = Agent(game)

		init_state = agent.get_array()
		init_act = random.randrange(4)

		runs += 1
		iters = 0

		# Run through environment
		while(1):

			# Get reward and next state and max - action
			game_state, reward = agent.take_step(init_act)
			# time.sleep(3)
			iters += 1

			if (game_state == -1 or game_state == 1):
				print reward
				print '{0:6d} {1:5d} {2:5d}'.format(game.score, game.max, iters)
				break

			next_state = np.append(agent.get_array(), [0])
			
			# Add data to mini-batch
			replays.add_instance(np.append(init_state, [init_act]), reward, next_state)

			# Run through replay - WTF?
			if (False and (replays.filled or replays.iter >= 512)):
				x,y,z = replays.sample()
				
				for replay_iter in xrange(replay_iters):
					q_max = -1000
					sim_state = z[replay_iter][:].reshape(16)
					sim_state = np.append(sim_state, 0)
					for a in xrange(4):
						if (game.can_move(a)):
							sim_state[16] = a
							pred = model.predict(sim_state.reshape([1, 17]))
							if (pred > q_max):
								q_max = pred
					y[replay_iter] += discount * q_max

				model.fit(x, y, nb_epoch=1, batch_size=bsize, verbose=0)

			# Switch init_* with epsilon greedy
			init_state = next_state[:16]
			
			init_act = 0
			max_so_far = -1000

			for a in xrange(4):
				if (game.can_move(a)):
					next_state[16] = a
					pred = model.predict(next_state.reshape([1, 17]))
					if (pred > max_so_far):
						max_so_far = pred
						init_act = a

			epsilon = max(((min_epsilon - max_epsilon)*runs)/runs_to_min_epsilon + max_epsilon, min_epsilon)
			if (random.random() < epsilon):
				num_moves = 0
				for a in xrange(4):
					if (game.can_move(a)):
						num_moves += 1
				seed = random.randrange(num_moves)
				for a in xrange(4):
					if (game.can_move(a)):
						if (seed == 0):
							init_act = a
						else:
							seed -= 1

		if (runs % save_stops == 0):

			model_json = model.to_json()
			with open("model.json", "w") as json_file:
			    json_file.write(model_json)
			model.save_weights("model.h5")


