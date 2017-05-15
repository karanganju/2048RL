import numpy as np
from envCNN import Env,Agent
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Conv2D, Activation, BatchNormalization
from keras.layers import Flatten
import keras
import random
import time
import sys, getopt
import os, pickle
from keras import backend as K

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],"l:",[])
    for opt, arg in opts:
        if opt == '-l':
            loadfile = arg

    model = Sequential()
        
    model.add(Conv2D(64, 3, 3, border_mode='same', init='uniform', name = 'conv1', input_shape=(4, 4, 1)))
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
    # model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    # json_file = open(loadfile +".json", 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    model.load_weights("models/sim_default/controlexps/model_SL_very_deep_bn_aug")
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    total_moves = 0
    total_score = 0
    total_tiles = 0

    for i in xrange(100):

        game = Env(False, 4)
        agent = Agent(game,False)
        steps = 0

        while(1):
            steps += 1
            state = agent.get_array()
            actions = model.predict(state.reshape([1, 4, 4, 1]))
            # print actions
            act = np.argmax(actions)

            # x = random.random()
            # # print x
            # if (x < 0.1):
            #     act = random.randrange(4)
                # print "Random bro!"
            # if (act == 0):
            #   print "Up"
            # elif (act == 1):
            #   print "Right"
            # elif (act == 2):
            #   print "Down"
            # elif (act == 3):
            #   print "Left"
	        
            # time.sleep(0.1)
            status, reward = agent.take_step(act)
            
            while (state == agent.get_array()).all():
                act = random.randrange(4)
                status, reward = agent.take_step(act)
            
            if(status != 0):
#                time.sleep(2)
#                print game.print_state()
                print '{0:6d} {1:5d} {2:5d}'.format(game.score, game.max, steps)
                total_moves += steps
                total_tiles += game.max
                total_score += game.score
                break
    
    print '{0:6d} {1:5d} {2:5d}'.format(total_score, total_tiles, total_moves)

