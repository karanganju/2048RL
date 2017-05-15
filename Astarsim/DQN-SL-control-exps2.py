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
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))

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

def create_model(supervised = False,finetuning = False, old_model = None):
    if (old_model is None and supervised == True):
        model = Sequential()
        model.add(Conv2D(64, 3, 3, border_mode='same', init='uniform', name = 'conv1', input_shape=(board_size, board_size, channel_size)))
        # model.add(BatchNormalization(name = 'bn1'))
        model.add(Activation('relu', name = 'rl1'))
        model.add(Conv2D(128, 3, 3, border_mode='same', init='uniform', name = 'conv2'))
        # model.add(BatchNormalization(name = 'bn2'))
        model.add(Activation('relu', name = 'rl2'))
        model.add(Conv2D(256, 3, 3, border_mode='same', init='uniform', name = 'conv3'))
        # model.add(BatchNormalization(name = 'bn3'))
        model.add(Activation('relu', name = 'rl3'))
        model.add(Flatten(name = 'fl'))
        model.add(Dense(64, init='uniform', name = 'fc1'))
        # model.add(BatchNormalization(name = 'bn4'))
        model.add(Activation('relu', name = 'rl4'))
        model.add(Dense(32, init='uniform', name = 'fc2'))
        # model.add(BatchNormalization(name = 'bn5'))
        model.add(Activation('relu', name = 'rl5'))
        model.add(Dense(16, init='uniform', name = 'fc3'))
        # model.add(BatchNormalization(name = 'bn6'))
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

    SL_states, SL_labels = cp.load(open('SL_data_512','rb'))
    SL_val_states, SL_val_labels = cp.load(open('SL_val_512','rb'))

    # history = supervised_model.fit(states, np.eye(4)[labels.astype('int')], batch_size=256, nb_epoch=1, validation_split=0.1, verbose=2, shuffle=True, callbacks=[keras.callbacks.EarlyStopping(min_delta=0.01, patience=5), keras.callbacks.ProgbarLogger()])
    history = supervised_model.fit(SL_states, np.eye(4)[SL_labels.astype('int')], 64, nb_epoch=256, verbose=2, validation_data=(SL_val_states, np.eye(4)[SL_val_labels.astype('int')]), shuffle=True, callbacks=[keras.callbacks.EarlyStopping(min_delta=0.0005, patience=5), keras.callbacks.ProgbarLogger(), keras.callbacks.ModelCheckpoint('models/sim_default/controlexps/model_SL_very_deep', 'val_loss', save_best_only=True)])

    # Save SL model
    # filename = 'models/sim_' + str(folder_num) + '/controlexps/model_SL_deep'
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
