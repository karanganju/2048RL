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
from simulation import *
import cPickle as cp

# Hyperparams
read_model = False
read_model_index = "sim_default/model_SL_augmax_bn_conv_large_actual"
discount = 0.99
max_epsilon = 0.4
min_epsilon = 0.1
replay_size = 2048*1024
replay_iters = 256
bsize = 256
save_stops = 50 #runs
runs = 0
runs_to_min_epsilon = 2048 #runs
copy_to_target_timeout = 2048 #steps
replay_start_train = 4096*2 #steps
test_on_holdout = True
holdout_size = 512
validation_timeout_runs = 20 #runs
board_size = 4 # Testing
handcrafted_features = False
input_size = board_size * board_size
channel_size = 1
not_changed = True
top_layer_checks = 1
SL_states = None
SL_labels = None

def generator(batch_size):
    # Create empty arrays to contain batch of features and labels#
    if (SL_states is None or SL_labels is None):
        raise ValueError('SL_states not available!')
    
    batch_states = None
    batch_labels = None

    while True:
        elems, counts = np.unique(np.random.randint(10, size=batch_size), return_counts=True)
        for i in xrange(len(elems)):
            samples = np.random.randint(np.shape(SL_states[elems[i]])[0], size=counts[i])
            if (batch_states is None):
                batch_states = SL_states[elems[i]][samples]
                batch_labels = SL_labels[elems[i]][samples]
            else :
                batch_states = np.append(batch_states, SL_states[elems[i]][samples], axis=0)
                batch_labels = np.append(batch_labels, SL_labels[elems[i]][samples])
        yield batch_states, np.eye(4)[batch_labels.astype('int')]

def create_model(supervised = False,finetuning = False, old_model = None):
    if (old_model is None and supervised == True):
        json_file = open("models/"+ str(read_model_index) +".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("models/" + str(read_model_index) +".h5")
        return model
    elif (supervised == False and not old_model is None):
        model = Sequential()
        model.add(Conv2D(128, 3, 3, border_mode='same', init='uniform', name = 'conv1', input_shape=(board_size, board_size, channel_size), trainable = not finetuning))
        model.add(BatchNormalization(name = 'bn1', trainable = not finetuning))
        model.add(Activation('relu', name = 'rl1', trainable = not finetuning))
        model.add(Conv2D(64, 3, 3, border_mode='same', init='uniform', name = 'conv2', trainable = not finetuning))
        model.add(BatchNormalization(name = 'bn2', trainable = not finetuning))
        model.add(Activation('relu', name = 'rl2', trainable = not finetuning))
        model.add(Flatten(name = 'fl', trainable = not finetuning))
        model.add(Dense(64, init='uniform', name = 'fc1', trainable = not finetuning))
        model.add(BatchNormalization(name = 'bn3', trainable = not finetuning))
        model.add(Activation('relu', name = 'rl3', trainable = not finetuning))
        model.add(Dense(32, init='uniform', name = 'fc2', trainable = not finetuning))
        model.add(BatchNormalization(name = 'bn4', trainable = not finetuning))
        model.add(Activation('relu', name = 'rl4', trainable = not finetuning))
        model.add(Dense(16, init='uniform', name = 'fc3'))
        model.add(BatchNormalization(name = 'bn5'))
        model.add(Activation('relu', name = 'rl5'))
        model.add(Dense(4, init='uniform'))
        model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

        model.get_layer('conv1').set_weights(old_model.get_layer('conv1').get_weights())
        model.get_layer('bn1').set_weights(old_model.get_layer('bn1').get_weights())
        model.get_layer('rl1').set_weights(old_model.get_layer('rl1').get_weights())
        model.get_layer('conv2').set_weights(old_model.get_layer('conv2').get_weights())
        model.get_layer('bn2').set_weights(old_model.get_layer('bn2').get_weights())
        model.get_layer('rl2').set_weights(old_model.get_layer('rl2').get_weights())
        model.get_layer('fc1').set_weights(old_model.get_layer('fc1').get_weights())
        model.get_layer('bn3').set_weights(old_model.get_layer('bn3').get_weights())
        model.get_layer('rl3').set_weights(old_model.get_layer('rl3').get_weights())
        model.get_layer('fc2').set_weights(old_model.get_layer('fc2').get_weights())
        model.get_layer('bn4').set_weights(old_model.get_layer('bn4').get_weights())
        model.get_layer('rl4').set_weights(old_model.get_layer('rl4').get_weights())
        return model

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

    def __init__(self, replay, validation_set, grad_desc_lr, old_model):
        
        self.replays = replay
        self.validation_set = validation_set
        self.learning_rate = grad_desc_lr
        self.model = create_model(False, True, old_model)
        self.target_model = create_model(False, True, old_model)

    def run_through_replay(self):
        if (self.replays.filled or self.replays.iter >= replay_start_train):
            s,a,r,s_new = replays.sample()
            y = np.zeros([self.replays.sample_size, 4])
            
            for replay_iter in xrange(self.replays.sample_size):
                sim_state = s_new[replay_iter]
                y[replay_iter] = self.model.predict(s[replay_iter].reshape([1, board_size, board_size, channel_size]))
                y[replay_iter][a[replay_iter]] = r[replay_iter] + discount * self.Q_max_from_target(sim_state)

            self.model.fit(s, y, nb_epoch=1, batch_size=bsize, verbose=0)
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
    folder_num = "RL"
    opts, args = getopt.getopt(sys.argv[1:],"i:l:s:",[])
    for opt, arg in opts:
        if opt == '-i':
            read_model = True
            read_model_index = arg
        elif opt == '-l':
            grad_desc_lr = arg
        elif opt == '-s':
            folder_num = arg

    # Supervised part
    supervised_model = create_model(True, False, None)
    supervised_runs = 0
    new_loss = 0.0
    old_loss = -np.inf

    # SL_states, SL_labels = cp.load(open('SL_data_sorted','rb'))

    # # history = supervised_model.fit(states, np.eye(4)[labels.astype('int')], batch_size=256, nb_epoch=1, validation_split=0.1, verbose=2, shuffle=True, callbacks=[keras.callbacks.EarlyStopping(min_delta=0.01, patience=5), keras.callbacks.ProgbarLogger()])
    # history = supervised_model.fit_generator(generator(64), samples_per_epoch=1024*512, nb_epoch=128, verbose=2, validation_data=generator(64), nb_val_samples=128, callbacks=[keras.callbacks.EarlyStopping(min_delta=0.001, patience=3), keras.callbacks.ProgbarLogger()])

    # Save SL model
    # filename = 'models/sim_' + str(folder_num) + '/model_SL'
    # json_filename = "{}.json".format(filename)
    # h5_filename = "{}.h5".format(filename)
    # model_json = supervised_model.to_json()
    # if not os.path.exists(os.path.dirname(json_filename)):
    #     try:
    #         os.makedirs(os.path.dirname(json_filename))
    #     except OSError as exc: # Guard against race condition
    #         if exc.errno != errno.EEXIST:
    #             raise
    # with open(json_filename, "w") as json_file:
    #     json_file.write(model_json)
    # supervised_model.save_weights(h5_filename)

    # RL Part
    replays = Replays(replay_size, replay_iters)

    # validation_set isn't meant to be sampled from -> hence, replay_iters = 0
    validation_set = Replays(holdout_size, 0)

    # Fill validation set
    # Needs to be chosen well - for now random
    if (test_on_holdout):
        fill_val_set(validation_set)
    
    dqn = DQN(replays, validation_set, grad_desc_lr, supervised_model)

    iters = 0
    prev_iters  = 0
    old_loss = -np.inf
    new_loss = 0.0
    top_layer_trained = 0

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
            # if (next_state != init_state).any():
            #     # Add all instances here.
            #     elem_min = 12
            #     elem_max = 0
            #     for i in xrange(4):
            #         for j in xrange(4):
            #             if (init_state[i][j] < elem_min):
            #                 elem_min = init_state[i][j]
            #             if (init_state[i][j] > elem_max):
            #                 elem_max = init_state[i][j]

            #     state_0 = np.copy(init_state)
            #     state_1 = np.copy(next_state)
            #     for i in xrange(1-elem_min, 10-elem_max):
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

        if (dqn.replays.filled or dqn.replays.iter >= replay_start_train):

            if (test_on_holdout and runs % validation_timeout_runs == 0):
                loss, Q_val = dqn.evaluate()
                print "Metrics (RL):", loss, ",",  Q_val[0]
                
                if(not_changed):
                    if (runs % (top_layer_checks * validation_timeout_runs) == 0 and runs > 0):
                        if (new_loss - old_loss < 0.01):
                            top_layer_trained += 1
                        else:
                            top_layer_trained = 0
                        old_loss = new_loss
                        new_loss = 0.0
                    
                    new_loss += loss

            if (top_layer_trained > 3 and not_changed):
                dqn.model = create_model(False, False, dqn.model)
                dqn.target_model = create_model(False, False, dqn.model)
                dqn.copy_to_target_model()
                not_changed = False
                runs = 0
                print "Unfreezing!"

            if (runs % save_stops == 0):
                filename = 'sim_' + str(folder_num) + '/model_{}'.format(runs/save_stops)
                dqn.save_model('models/{}'.format(filename))

