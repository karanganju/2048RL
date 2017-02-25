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
		self.x = np.zeros([self.cap, 17], dtype=int)
		self.y = np.zeros([self.cap, 1], dtype=float)

	def sample(self):
		ret_x = np.zeros([self.sample_size, 17])
		ret_y = np.zeros([self.sample_size])

		for i in xrange(self.sample_size):
			r = 0
			if (self.filled):
				r = random.randrange(self.cap)
			else:
				r = random.randrange(self.iter+1)

			for j in xrange(17):
				ret_x[i][j] = self.x[r][j]
			ret_y[i] = self.y[r]

		return ret_x, ret_y

	def add_instance(self, arr, y):
		self.iter = self.iter+1
		if (self.iter == self.cap):
			self.filled = True
			self.iter = 0
		for i in xrange(17):
			self.x[self.iter][i] = arr[i]
		# Take s,a instead of R+gamma*Q ?
		self.y[self.iter] = y


if __name__ == '__main__':

	read_model = False

	opts, args = getopt.getopt(sys.argv,"i",[])
	for opt, arg in opts:
		if opt == '-i':
			read_model = True

	# Hyperparams
	discount = 0.9
	epsilon = 0.1
	replay_size = 1024
	replay_iters = 32
	save_stops = 50
	runs = 0

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
		model.add(Dense(17, input_dim=17, init='uniform', activation='sigmoid'))
		model.add(Dense(9, init='uniform', activation='sigmoid'))
		model.add(Dense(9, init='uniform', activation='sigmoid'))
		model.add(Dense(9, init='uniform', activation='sigmoid'))
		model.add(Dense(1, init='uniform', activation='sigmoid'))

		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])


	while(1):

		game = Env(False)
		agent = Agent(game)
		replays = Replays(replay_size, replay_iters)

		init_state = agent.get_array()
		init_act = random.randrange(4)

		runs += 1
		iters = 0

		# Run through environment
		while(1):
			# Get reward and next state and max - action
			reward = agent.take_step(init_act)
			# time.sleep(3)
			iters += 1

			if (reward == 1000 or reward == -100):
				print '{0:6d} {1:5d} {2:5d}'.format(game.score, game.max, iters)
				break

			next_state = np.append(agent.get_array(), [0])
			max_so_far = -1000
			next_act = 0
			for a in xrange(4):
				if (game.can_move(a)):
					next_state[16] = a
					pred = model.predict(next_state.reshape([1, 17]))
					if (pred > max_so_far):
						next_act = a
						max_so_far = pred

			# Update model
			model.fit(np.append(init_state, [init_act]).reshape([1,17]), reward+discount*max_so_far, nb_epoch=1, batch_size=1, verbose=0)

			# Add data to mini-batch
			replays.add_instance(np.append(init_state, [init_act]), reward+discount*max_so_far)

			# Run through replay
			if (iters % 4 == 3):
				x,y = replays.sample()
				model.fit(x, y, nb_epoch=1, batch_size=replay_iters, verbose=0)

			# Maybe add target network

			# Switch init_* with epsilon greedy
			init_state = next_state[:16]
			init_act = next_act
			if (random.random() < epsilon/runs):
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


