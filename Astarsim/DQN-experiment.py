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
board_size = 4
channel_size = 12

if __name__ == '__main__':
    
    model = Sequential()
    model.add(Flatten(input_shape=(board_size, board_size, channel_size)))
    model.add(Dense(32, init='uniform', activation='relu'))
    model.add(Dense(32, init='uniform', activation='relu'))
    model.add(Dense(16, init='uniform', activation='relu'))
    model.add(Dense(4, init='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

    while(1):
        states, labels = simulation(0)

        for i in xrange(1):
            x, y = simulation(0)
            states = np.append(states, x, axis = 0)
            labels = np.append(labels, y)

        permutation = np.random.permutation(states.shape[0])

        model.fit(states[permutation], np.eye(4)[labels[permutation].astype('int')], batch_size=256, nb_epoch=2, validation_split=0.05, verbose=2)