# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
import numpy as np
from env import Env,Agent
from keras.models import Sequential
from keras.layers import Dense
import random
import time
import sys, getopt


# Hyperparams
read_model = False
read_model_index = 1
discount = 0.99
max_epsilon = 1
min_epsilon = 0.1
replay_size = 2048*256
replay_iters = 128
bsize = 128
save_stops = 100
runs = 0
runs_to_min_epsilon = 2048
copy_to_target_timeout = 1024
replay_start_train = 4096*2

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
		ret_s = np.zeros([self.sample_size, 16], dtype=int)
		ret_a = np.zeros([self.sample_size], dtype=int)
		ret_r = np.zeros([self.sample_size])
		ret_s_new = np.zeros([self.sample_size, 16], dtype=int)

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
	def add_instance(self, init_state, action, reward, next_state):
		self.iter = self.iter+1
		if (self.iter == self.cap):
			self.filled = True
			self.iter = 0
		self.s[self.iter] = init_state[0]
		self.s_new[self.iter] = next_state[0]
		self.r[self.iter] = reward
		self.a[self.iter] = action

class DQN(object):

	def __init__(self, replay):
		
		self.replays = replay
		self.model = self.create_model()
		self.target_model = self.create_model()

	def create_model(self):
		if (read_model):
			json_file = open("models/model_"+ str(read_model_index) +".json", 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			model = model_from_json(loaded_model_json)
			# load weights into new model
			model.load_weights("models/model_"+ str(read_model_index) +".h5")
		else:
			model = Sequential()
			model.add(Dense(16, input_dim=16, init='uniform', activation='relu'))
			model.add(Dense(8, init='uniform', activation='relu'))
			model.add(Dense(8, init='uniform', activation='relu'))
			model.add(Dense(8, init='uniform', activation='relu'))
			model.add(Dense(4, init='uniform'))
			model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
		return model

	def run_through_replay(self):
		if (self.replays.filled or self.replays.iter >= replay_start_train):
			s,a,r,s_new = replays.sample()
			y = np.zeros([self.replays.sample_size, 4])
			
			for replay_iter in xrange(self.replays.sample_size):
				sim_state = s_new[replay_iter].reshape([1,16])
				y[replay_iter] = self.target_model.predict(s[replay_iter].reshape([1,16]))
				y[replay_iter][a[replay_iter]] = r[replay_iter] + discount * self.Q_max(sim_state)

			self.model.fit(s, y, nb_epoch=1, batch_size=bsize, verbose=0)

	def select_epsilon_greedy(self, state):
		act_taken = self.select_greedy(state)
		epsilon = max(((min_epsilon - max_epsilon)*runs)/runs_to_min_epsilon + max_epsilon, min_epsilon)

		if (random.random() < epsilon):
			act_taken = random.randrange(4)

		return act_taken

	def select_greedy(self, state):
		Q_vals = self.model.predict(state)
		return np.argmax(Q_vals)

	def Q_max(self, state):
		Q_vals = self.model.predict(state)
		return np.max(Q_vals)

	def copy_to_target_model(self):

		self.target_model.set_weights(self.model.get_weights())

	def save_model(self, index):
		model_json = self.model.to_json()
		with open("models/model_"+ str(index) +".json", "w") as json_file:
		    json_file.write(model_json)
		self.model.save_weights("models/model_"+ str(index) +".h5")

if __name__ == '__main__':

	opts, args = getopt.getopt(sys.argv,"i:",[])
	for opt, arg in opts:
		if opt == '-i':
			read_model = True

	replays = Replays(replay_size, replay_iters)
	dqn = DQN(replays)
	iters = 0
	prev_iters  = 0

	while(1):

		game = Env(False)
		agent = Agent(game)

		init_state = agent.get_array()
		# Epsilon greedy here.
		init_act = dqn.select_epsilon_greedy(init_state)

		runs += 1
		# iters = 0

		# Run through environment
		while(1):

			if (iters % copy_to_target_timeout == 0):
				dqn.copy_to_target_model()

			# Get reward and next state and max - action
			game_state, reward = agent.take_step(init_act)
			# time.sleep(3)
			iters += 1

			next_state = agent.get_array()
			
			# Add data to mini-batch
			replays.add_instance(init_state, init_act, reward, next_state) 

			# Run through replay
			dqn.run_through_replay()
			
			# Game over
			if (game_state == -1 or game_state == 1):
				print '{0:6d} {1:5d} {2:5d}'.format(game.score, game.max, iters - prev_iters)
				prev_iters = iters
				break

			# Switch init_* with epsilon greedy
			init_state = next_state
			init_act = dqn.select_epsilon_greedy(init_state)
			
		if (runs % save_stops == 0):
			dqn.save_model(runs/save_stops)
