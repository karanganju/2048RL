import cPickle as cp
import numpy as np

states, labels = cp.load(open('SL_data', 'rb'))

if (states is None or labels is None):
    raise ValueError('states not available!')

# states = states[:282963,:,:,:]
# labels = labels[:282963]

# size = np.shape(states)[0]

# new_states = np.zeros([8*size,4,4,1])
# new_labels = np.zeros([8*size,])

# # Data Aug #1
# for i in xrange(size):
#     new_states[i*8] = states[i]
#     new_labels[i*8] = labels[i]
#     new_states[i*8 + 1] = np.fliplr(states[i])
#     if (labels[i] % 2 == 1):
#         new_labels[i*8 + 1] = (labels[i] + 2) % 4
#     else:
#         new_labels[i*8 + 1] = labels[i]

#     for j in xrange(1,4):
#         for k in xrange(2):
#             new_states[i*8 + j*2 + k] = np.rot90(new_states[i*8 + j*2 - 2 + k])
#             new_labels[i*8 + j*2 + k] = (new_labels[i*8 + j*2 - 2 + k] + 3) % 4

cp.dump((states, labels), open('SL_data_512','wb'))