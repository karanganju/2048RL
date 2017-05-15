import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Conv2D
# from keras.layers import Flatten
# import keras
# import random
# from keras import backend as K
from simulation import *
import cPickle as cp

# Experiment for Supervised Imitation Learning
board_size = 4
channel_size = 12

if __name__ == '__main__':
    
	SL_states, SL_labels = cp.load(open('SL_data_small_sorted','rb'))
	states = SL_states[0]
	labels = SL_labels[0]
	for i in xrange(1,10):
		states = np.append(states, SL_states[i], axis=0)
		labels = np.append(labels, SL_labels[i])

	state_label_concat = np.concatenate((np.reshape(states, [np.shape(states)[0], 16]), np.reshape(labels, [np.shape(states)[0], 1])), axis=1)

	exit()
    # while(1):
    # 	states, labels = simulation(0)
    #     x, y = simulation(0)
    #     states = np.append(states, x, axis = 0)
    #     labels = np.append(labels, y)

    #     print np.shape(states)[0]

    #     if (np.shape(states)[0] > 100000):
    #         cp.dump((states, labels), open('SL_data_small', 'wb'))
    #         break