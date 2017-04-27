#!/usr/bin/env python
# Copyright 2014 Google Inc. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import os
import sys
import time
import random
import multiprocessing
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

#from telemetry.core import browser_finder
#from telemetry.core import browser_options
from kcwu import *

KEY_CODE = {'left': 37,
            'up': 38,
            'right': 39,
            'down': 40}

K_CODE2 = {'left' : 3,
            'up': 0,
            'right': 1,
            'down': 2}

NCPU = 1
#NCPU = 12
ITERATION = 100
#ITERATION = 30
#ITERATION = 12
#ITERATION = 1

from game2048 import GameManager

class Dummy:
  def write(self, s):
    pass
  def flush(self):
    pass

remain = multiprocessing.Value('i')
timeout_count = multiprocessing.Value('i')
def simulation(idx):
  states = None
  ret_vals = np.array([])

  random.seed(idx)
  if idx > 0:
    sys.stdout = Dummy()

  gm = GameManager()

  step = 0
  total_time = 0
  stale_steps = 0
  grid = None
  last_grid = None
  times = []
  while not gm.isOver():
    step += 1
    last_grid = grid
    grid = gm.getGrid()
    if grid == last_grid:
      stale_steps += 1
    else:
      stale_steps = 0
    if stale_steps >= 10:
      sys.stderr.write('stale idx=%d\n' % idx)
      assert 0
      timeout_count.value = -99999
    nextKey = gm.ai.getNextMove(grid)
    # if t1 - t0 > 0.1:
    #   timeout_count.value += 1
    #   sys.stderr.write('t %f, count=%d\n' % (t1 - t0, timeout_count.value))
    # print '%r : %d' % (gm.getGridFormatted(), K_CODE2[nextKey])
    grid_arr = np.array(gm.board.board)
    for i in xrange(4):
      for j in xrange(4):
        if (grid_arr[i][j] == None):
          grid_arr[i][j] = 1

    # print np.eye(12)[np.log2(grid_arr.astype('float')).astype(int)]
    if (states is None):
      states = [np.eye(12)[np.log2(grid_arr.astype('float')).astype(int)]]
    else :
      states = np.append(states, [np.eye(12)[np.log2(grid_arr.astype('float')).astype(int)]], axis = 0)
    ret_vals = np.append(ret_vals, np.array([K_CODE2[nextKey]]))

    gm.pressKey(KEY_CODE[nextKey])
    # gm.board.show()
    for m in KEY_CODE.keys():
      if gm.board.canMove(gm.getGrid(), m):
        break
    else:
      break
    #time.sleep(0.03)
    if gm.isWin():
      return states, ret_vals

  return states, ret_vals

def Main(args):
  while(1):
    simulation(0)
  return 0


if __name__ == '__main__':
  sys.exit(Main(sys.argv[1:]))
# vim:sw=2:expandtab
