import numpy as np
from envCNN import Env,Agent
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
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
read_model_index = 1
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
SL_val_states = None
SL_val_labels = None
folder_num = "default"

def generator(batch_size, val = False):
    # Create empty arrays to contain batch of features and labels#
    if (val):
        states = SL_val_states
        labels = SL_val_labels
    else:
        states = SL_states
        labels = SL_labels

    if (states is None or labels is None):
        raise ValueError('states not available!')
    
    batch_states = None
    batch_labels = None

    while True:
        elems, counts = np.unique(np.random.randint(10, size=batch_size), return_counts=True)
        for i in xrange(len(elems)):
            samples = np.random.randint(np.shape(states[elems[i]])[0], size=counts[i])
            if (batch_states is None):
                batch_states = states[elems[i]][samples]
                batch_labels = labels[elems[i]][samples]
            else :
                batch_states = np.append(batch_states, states[elems[i]][samples], axis=0)
                batch_labels = np.append(batch_labels, labels[elems[i]][samples])
        
        # Data Aug #1
        for i in xrange(np.shape(batch_states)[0]):
            x = random.random()
            if (x <= 0.5):
                batch_states[i] = np.fliplr(batch_states[i])
                if (batch_labels[i] % 2 == 1):
                    batch_labels[i] = (batch_labels[i] + 2) % 4
            x = random.random()
            if (x < 0.25):
                batch_states[i] = np.rot90(batch_states[i])
                batch_labels[i] = (batch_labels[i] + 3) % 4
            if (x < 0.5):
                batch_states[i] = np.rot90(batch_states[i])
                batch_labels[i] = (batch_labels[i] + 3) % 4
            if (x < 0.75):
                batch_states[i] = np.rot90(batch_states[i])
                batch_labels[i] = (batch_labels[i] + 3) % 4

        # Data Aug #2
	   for i in xrange(np.shape(batch_states)[0]):
            elem_min = 12
            elem_max = 0
            for x in xrange(4):
                for y in xrange(4):
                    if not (batch_states[i][x][y][0] == 0):
                        if(batch_states[i][x][y][0] < elem_min):
                            elem_min = batch_states[i][x][y][0]
                        if(batch_states[i][x][y][0] > elem_max):
                            elem_max = batch_states[i][x][y][0]
            inc = random.randrange(1-elem_min, 11-elem_max)
            for x in xrange(4):
                for y in xrange(4):
                    if not (batch_states[i][x][y][0] == 0):
                        batch_states[i][x][y][0] += inc
        yield batch_states, np.eye(4)[batch_labels.astype('int')]

def create_model(supervised = False,finetuning = False, old_model = None):
    if (old_model is None and supervised == True):
        model = Sequential()
        model.add(Conv2D(64, 3, 3, border_mode='same', init='uniform', name = 'conv1', input_shape=(board_size, board_size, channel_size)))
        model.add(BatchNormalization(name = 'bn1'))
        model.add(Activation('relu', name = 'rl1'))
        model.add(Conv2D(128, 3, 3, border_mode='same', init='uniform', name = 'conv2'))
        model.add(BatchNormalization(name = 'bn2'))
        model.add(Activation('relu', name = 'rl2'))
        model.add(Conv2D(256, 3, 3, border_mode='same', init='uniform', name = 'conv3'))
        model.add(BatchNormalization(name = 'bn3'))
        model.add(Activation('relu', name = 'rl3'))
        model.add(Flatten(name = 'fl'))
        model.add(Dense(64, init='uniform', name = 'fc1'))
        model.add(BatchNormalization(name = 'bn4'))
        model.add(Activation('relu', name = 'rl4'))
        model.add(Dense(32, init='uniform', name = 'fc2'))
        model.add(BatchNormalization(name = 'bn5'))
        model.add(Activation('relu', name = 'rl5'))
        model.add(Dense(16, init='uniform', name = 'fc3'))
        model.add(BatchNormalization(name = 'bn6'))
        model.add(Activation('relu', name = 'rl6'))
        model.add(Dense(4, init='uniform', activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        return model
    
if __name__ == '__main__':
    # Supervised part
    supervised_model = create_model(True, False, None)
    supervised_runs = 0
    new_loss = 0.0
    old_loss = -np.inf

    SL_states, SL_labels = cp.load(open('SL_data_aug_512','rb'))
    SL_val_states, SL_val_labels = cp.load(open('SL_val_512','rb'))

    # history = supervised_model.fit(states, np.eye(4)[labels.astype('int')], batch_size=256, nb_epoch=1, validation_split=0.1, verbose=2, shuffle=True, callbacks=[keras.callbacks.EarlyStopping(min_delta=0.01, patience=5), keras.callbacks.ProgbarLogger()])
    history = supervised_model.fit_generator(generator(64), samples_per_epoch=1024*1024, nb_epoch=128, verbose=2, validation_data=generator(64, True), nb_val_samples=2048, pickle_safe=False, callbacks=[keras.callbacks.EarlyStopping(min_delta=0.0005, patience=5), keras.callbacks.ProgbarLogger()])

    # Save SL model
    filename = 'models/sim_' + str(folder_num) + '/controlexps/model_SL_deep'
    json_filename = "{}.json".format(filename)
    h5_filename = "{}.h5".format(filename)
    model_json = supervised_model.to_json()
    if not os.path.exists(os.path.dirname(json_filename)):
        try:
            os.makedirs(os.path.dirname(json_filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(json_filename, "w") as json_file:
        json_file.write(model_json)
    supervised_model.save_weights(h5_filename)
