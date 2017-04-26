#!/usr/bin/env python
# Copyright 2014 Google Inc. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import os
import sys
import time
import random
import multiprocessing

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
    t0 = time.time()
    nextKey = gm.ai.getNextMove(grid)
    t1 = time.time()
    total_time += t1 - t0
    times.append(t1 - t0)
    times.sort(reverse=True)
    times = times[:20]
    # if t1 - t0 > 0.1:
    #   timeout_count.value += 1
    #   sys.stderr.write('t %f, count=%d\n' % (t1 - t0, timeout_count.value))
    gm.board.show()
    print '%r : %d' % (gm.getGrid(), K_CODE2[nextKey])
    print nextKey
    gm.pressKey(KEY_CODE[nextKey])
    # gm.board.show()
    for m in KEY_CODE.keys():
      if gm.board.canMove(gm.getGrid(), m):
        break
    else:
      break
    #time.sleep(0.03)
    if gm.isWin():
      sys.exit()

  remain.value -= 1
  times = [int(t*1000) for t in times]
  sys.stderr.write('max times %r\n' % times)
  sys.stderr.write('%d score %d\n' % (idx, gm.getScore()))
  sys.stderr.write('simulation remain %d\n' % remain.value)
  sys.stdout.flush()

  return gm.getScore(), step, total_time

def Main(args):
  scores = []
  #options = browser_options.BrowserFinderOptions()
  #parser = options.CreateParser('telemetry_perf_test.py')
  #options, args = parser.parse_args(args)
  global ITERATION, NCPU
  if args:
    ITERATION = int(args[0])
  if len(args) >= 2:
    NCPU = int(args[1])
  NCPU = min(NCPU, ITERATION)

  #browser_to_create = browser_finder.FindBrowser(options)
  #assert browser_to_create

  #with browser_to_create.Create() as b:
  if 1:
    total_step = 0
    total_t = 0
    scores = []

    remain.value = ITERATION
    stdout = sys.stdout
    if NCPU == 1:
      result = map(simulation, range(ITERATION))
    else:
      pool = multiprocessing.Pool(processes=min(NCPU, ITERATION))
      result = pool.map(simulation, range(ITERATION))
    for score, step, t in result:
      scores.append(score)
      total_t += t
      total_step += step
    scores.sort()
    sys.stdout = stdout
    print "Scores = %r" % scores
    print "Avg = %f" % ((sum(scores) - max(scores) - min(scores)) /
        (ITERATION - 2.0))
    print '%f ms/step' % (1000.0*(total_t)/total_step)
    print 'timeout count', timeout_count.value

  return 0


if __name__ == '__main__':
  sys.exit(Main(sys.argv[1:]))
# vim:sw=2:expandtab
