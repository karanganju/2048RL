import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
import keras
import random
from keras import backend as K
from simulation import *

# Experiment for Supervised Imitation Learning

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

if __name__ == '__main__':
    
    model = Sequential()
    model.add(Dense(32, init='uniform', activation='relu', input_shape=(board_size, board_size, channel_size)))
    model.add(Flatten())
    model.add(Dense(32, init='uniform', activation='relu'))
    model.add(Dense(16, init='uniform', activation='relu'))
    model.add(Dense(4, init='uniform'))
    model.compile(loss='categorical_crossentropy', optimizer="rmsprop")

    while(1):
        states, labels = simulation(0)

        for i in xrange(1):
            x, y = simulation(0)
            states = np.append(states, x, axis = 0)
            labels = np.append(labels, y)

        print np.shape(states), np.shape(labels)

        permutation = np.random.permutation(states.shape[0])

        model.fit(states[permutation], np.eye(4)[labels[permutation].astype('int')], batch_size=128, nb_epoch=3, validation_split=0.05, )