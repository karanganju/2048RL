import numpy as np
from envCNN import Env,Agent
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
import keras
import random
import time
import sys, getopt
import os, pickle
from keras import backend as K

# Hyperparams
read_model = False
read_model_index = 1
discount = 0.99
max_epsilon = 1
min_epsilon = 0.1
replay_size = 2048*256
replay_iters = 512
bsize = 512
save_stops = 150 #runs
runs = 0
runs_to_min_epsilon = 2048 #runs
copy_to_target_timeout = 2048 #steps
replay_start_train = 4096*2 #steps
test_on_holdout = True
holdout_size = 512
validation_timeout_runs = 50 #runs
board_size = 4 # Testing
handcrafted_features = False
input_size = board_size * board_size
channel_size = 12


class Replays(object):

    def __init__(self, size, ssize):
        self.cap = size
        self.sample_size = ssize
        self.iter = -1
        self.filled = False
        self.s = np.zeros([self.cap, board_size, board_size, channel_size], dtype=int)
        self.a = np.zeros([self.cap, 1], dtype=int)
        self.r = np.zeros([self.cap, 1], dtype=float)
        self.s_new = np.zeros([self.cap, board_size, board_size, channel_size], dtype=int)

    def sample(self):
        ret_s = np.zeros([self.sample_size, board_size, board_size, channel_size], dtype=int)
        ret_a = np.zeros([self.sample_size], dtype=int)
        ret_r = np.zeros([self.sample_size], dtype=float)
        ret_s_new = np.zeros([self.sample_size, board_size, board_size, channel_size], dtype=int)

        for i in xrange(self.sample_size):
            r = 0
            if (self.filled):
                r = random.randrange(self.cap)
            else:
                r = random.randrange(self.iter+1)

            ret_s[i] = self.s[r]
            ret_s_new[i] = self.s_new[r]
            ret_r[i] = self.r[r]
            ret_a[i] = self.a[r]

        return ret_s, ret_a, ret_r, ret_s_new

    def add_instance(self, init_state, action, reward, next_state):
        self.iter = self.iter+1
        if (self.iter == self.cap):
            self.filled = True
            self.iter = 0
        self.s[self.iter] = init_state
        self.s_new[self.iter] = next_state
        self.r[self.iter] = reward
        self.a[self.iter] = action

class DQN(object):

    def __init__(self, replay, validation_set, grad_desc_lr=0.001):
        
        self.replays = replay
        self.validation_set = validation_set
        self.learning_rate = grad_desc_lr
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
            
            # model.add(Conv2D(64, 3, 3, activation='relu', input_shape=(board_size, board_size, channel_size), init='uniform', border_mode='same'))
            # model.add(Conv2D(64, 3, 3, init='uniform', activation='relu', border_mode='same'))
            # model.add(Flatten())
            
            model.add(Dense(32, init='uniform', activation='relu', input_shape=(board_size, board_size, channel_size)))

            model.add(Dense(32, init='uniform', activation='relu'))
            model.add(Dense(16, init='uniform', activation='relu'))
            model.add(Dense(4, init='uniform'))
            # model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
            model.compile(loss='mean_squared_error', optimizer="rmsprop")
        return model

    def run_through_replay(self):
        if (self.replays.filled or self.replays.iter >= replay_start_train):
            s,a,r,s_new = replays.sample()
            y = np.zeros([self.replays.sample_size, 4])
            
            for replay_iter in xrange(self.replays.sample_size):
                sim_state = s_new[replay_iter]
                y[replay_iter] = self.model.predict(s[replay_iter].reshape([1, board_size, board_size, channel_size]))
                y[replay_iter][a[replay_iter]] = r[replay_iter] + discount * self.Q_max_from_target(sim_state)

            self.model.fit(s, y, nb_epoch=10, batch_size=bsize, verbose=0)
            # print self.model.get_weights()[0][0]
            # print "END"

    def select_epsilon_greedy(self, state):
        act_taken = self.select_greedy(state)
        epsilon = max(((min_epsilon - max_epsilon)*runs)/runs_to_min_epsilon + max_epsilon, min_epsilon)

        if (random.random() < epsilon):
            act_taken = random.randrange(4)

        return act_taken

    def select_greedy(self, state):
        Q_vals = self.model.predict(state.reshape([1, board_size, board_size, channel_size]))
        return np.argmax(Q_vals)

    def Q_max_from_target(self, state):
        Q_vals = self.target_model.predict(state.reshape([1, board_size, board_size, channel_size]))
        return np.max(Q_vals)

    def copy_to_target_model(self):

        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, filename):
        json_filename = "{}.json".format(filename)
        h5_filename = "{}.h5".format(filename)
        model_json = self.model.to_json()
        if not os.path.exists(os.path.dirname(json_filename)):
            try:
                os.makedirs(os.path.dirname(json_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(json_filename, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(h5_filename)


    def evaluate(self):
        s = self.validation_set.s
        a = self.validation_set.a
        r = self.validation_set.r
        s_new = self.validation_set.s_new
        y = np.zeros([self.validation_set.cap, 4])

        Q_ave = 0.0

        for validation_iter in xrange(self.validation_set.cap):
            sim_state = s_new[validation_iter]
            y[validation_iter] = self.model.predict(s[validation_iter].reshape([1, board_size, board_size, channel_size]))
            Q_ave += y[validation_iter][a[validation_iter]]
            y[validation_iter][a[validation_iter]] = r[validation_iter] + discount*self.Q_max_from_target(sim_state)
        
        return self.model.evaluate(s, y, batch_size=self.validation_set.cap, verbose=0), Q_ave/self.validation_set.cap

def fill_val_set(val_set):
    for v_iter in xrange(holdout_size):
        game = Env(False, board_size)
        agent = Agent(game,handcrafted_features)
        while(1):
            init_state = agent.get_array()
            init_act = random.randrange(4)
            game_state, reward = agent.take_step(init_act)
            next_state = agent.get_array()
            if (random.random() < 0.01):
                val_set.add_instance(init_state, init_act, reward, next_state)
                break
            if (game_state != 0):
                v_iter -= 1
                break

if __name__ == '__main__':
    grad_desc_lr = 0.001
    folder_num = "default"
    opts, args = getopt.getopt(sys.argv[1:],"i:l:s:",[])
    for opt, arg in opts:
        if opt == '-i':
            read_model = True
            read_model_index = arg
        elif opt == '-l':
            grad_desc_lr = arg
        elif opt == '-s':
            folder_num = arg

    replays = Replays(replay_size, replay_iters)

    # validation_set isn't meant to be sampled from -> hence, replay_iters = 0
    validation_set = Replays(holdout_size, 0)

    # Fill validation set
    # Needs to be chosen well - for now random
    if (test_on_holdout):
        fill_val_set(validation_set)
    
    dqn = DQN(replays, validation_set, grad_desc_lr)

    iters = 0
    prev_iters  = 0

    while(1):

        game = Env(False, board_size)
        agent = Agent(game,handcrafted_features)

        init_state = agent.get_array()
        # Epsilon greedy here.
        init_act = dqn.select_epsilon_greedy(init_state)
        next_state = 0

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

            # Add data to mini-batch only if next state is different
            if (next_state != init_state).any():
                replays.add_instance(init_state, init_act, reward, next_state) 

            # Run through replay
            dqn.run_through_replay()

            # Game over
            if (game_state == -1 or game_state == 1):
                print '{0:6d} {1:5d} {2:5d}'.format(game.score, game.max, iters - prev_iters)
                prev_iters = iters
                break

            # Switch init_* with epsilon greedy
            init_state = np.copy(next_state)
            init_act = dqn.select_epsilon_greedy(init_state)

        if (runs % save_stops == 0):
            filename = 'sim_' + str(folder_num) + '/model_{}'.format(runs/save_stops)
            dqn.save_model('models/{}'.format(filename))

        # if (runs % (save_stops*20) == 0):
        #     filename = 'sim_'+folder_num + '/model_{}'.format(runs/save_stops)
        #     pickle_filename = 'replays/{}.pckl'.format(filename)
        #     print(os.path.dirname(pickle_filename))
        #     if not os.path.exists(os.path.dirname(pickle_filename)):
        #         try:
        #             os.makedirs(os.path.dirname(pickle_filename))
        #         except OSError as exc: # Guard against race condition
        #             if exc.errno != errno.EEXIST:
        #                 raise
        #     f = open(pickle_filename, 'wb')
        #     pickle.dump(dqn.replays, f)
        #     f.close()

        if (test_on_holdout and runs % validation_timeout_runs == 0):
            loss, Q_val = dqn.evaluate()
            print "Metrics:", loss, ",",  Q_val[0]