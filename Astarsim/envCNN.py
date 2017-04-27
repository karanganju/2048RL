import numpy as np
import random
import time
import prettytable
import math

def print_table(array, len):
	t = prettytable.PrettyTable(header=False, hrules=prettytable.ALL, border=True)
	for i in xrange(len):
		t.add_row(array[i])
	print t

class Env(object):

	def __init__(self, _is_human = False, sq_len = 4):
		# self.array = [[0,0,0,0],[8,4,4,0],[0,0,0,0],[0,0,0,0]]
		self.len = sq_len
		self.size = sq_len * sq_len
		self.array = np.zeros((self.len,self.len), dtype=int)
		self.is_human = _is_human;
		self.score = 0;
		self.unfilled = self.size;
		self.valid_move = False;
		self.has_merged = np.zeros(self.size)
		self.game_won = False
		self.max = 2
		# self.pp = pprint.PrettyPrinter(indent=10)
		self.take_step(debug_print = False);
		self.take_step();

	def game_lost(self):

		if self.unfilled > 0:
			return False

		for i in xrange(self.len):
			if (self.can_move(i)):
				return False

		return True


	def can_move(self, step):
		if (step == 0):
			for x in xrange(self.len - 1):
				for y in xrange(self.len):
					if(self.array[x+1][y] != 0 and (self.array[x+1][y] == self.array[x][y] or self.array[x][y] == 0)):
						return True

		# Right
		elif (step == 1):
			for y in xrange(self.len - 1):
				for x in xrange(self.len):
					if(self.array[x][self.len - 2 -y] != 0 and (self.array[x][self.len - 2 -y] == self.array[x][self.len - 1-y] or self.array[x][self.len - 1-y] == 0)):
						return True

		# Down
		elif (step == 2):
			for x in xrange(self.len - 1):
				for y in xrange(self.len):
					if(self.array[self.len-2-x][y] != 0 and (self.array[self.len-2-x][y] == self.array[self.len - 1-x][y] or self.array[self.len - 1-x][y] == 0)):
						return True

		# Left
		elif (step == 3):
			for y in xrange(self.len - 1):
				for x in xrange(self.len):
					if(self.array[x][y+1] != 0 and (self.array[x][y+1] == self.array[x][y] or self.array[x][y] == 0)):
						return True

		return False

	def take_step(self, step = -1, debug_print = True):

		debug_print = debug_print and self.is_human;
		self.valid_move = False;
		self.has_merged = np.zeros(self.size)

		# Might need to rewrite bottom part for readability

		# Up
		if (step == 0):
			for x in xrange(self.len - 1):
				for y in xrange(self.len):
					for steps in xrange(x+1):
						if(self.move(x+1-steps,y,x-steps,y) == -1):
							break

		# Right
		elif (step == 1):
			for y in xrange(self.len - 1):
				for x in xrange(self.len):
					for steps in xrange(y+1):
						if(self.move(x,self.len-2+steps-y,x,self.len - 1+steps-y) == -1):
							break

		# Down
		elif (step == 2):
			for x in xrange(self.len - 1):
				for y in xrange(self.len):
					for steps in xrange(x+1):
						if(self.move(self.len-2+steps-x,y,self.len - 1+steps-x,y) == -1):
							break

		# Left
		elif (step == 3):
			for y in xrange(self.len - 1):
				for x in xrange(self.len):
					for steps in xrange(y+1):
						if(self.move(x,y+1-steps,x,y-steps) == -1):
							break

		if (self.game_won):
			return 1

		if (self.valid_move or step == -1):
			filling_cell = random.randrange(self.unfilled);
			for x in xrange(self.size):
				if (self.array[x/self.len][x%self.len] == 0):
					if (filling_cell == 0):
						r = random.random()
						if (r < 0.9):
							self.array[x/self.len][x%self.len] = 2
						else:
							self.array[x/self.len][x%self.len] = 4
						self.unfilled-=1;
						break
					else:
						filling_cell-=1;

		if (debug_print):
			self.print_state()

		if (self.game_lost() == True):
			if (debug_print):
				print "GAME OVER"
			return -1
		else:
			return 0

	# Merge i,j to i2,j2
	def move(self, i, j, i2, j2):

		# Nothing to move
		if (self.array[i][j] == 0):
			return -1

		# Something has already merged with i2,j2 -> Multiple merges not allowed. Why no check for i,j? :
		# because of the way we are traversing \_(-_-)_/
		elif (self.has_merged[i2*self.len+j2] == 1):
			return -1

		elif (self.array[i2][j2] == 0 or self.array[i2][j2] == self.array[i][j]):
			ret_val = 0
			self.score += 2*self.array[i2][j2];
			if (self.array[i2][j2] != 0):
				self.unfilled+=1;
				ret_val = -1;
				self.has_merged[i2*self.len+j2] = 1
			self.array[i2][j2] = self.array[i][j] + self.array[i2][j2];
			self.array[i][j] = 0;
			self.max = max(self.max, self.array[i2][j2])
			if (self.array[i2][j2] == 2048):
				self.game_won = True
			self.valid_move = True;
			return ret_val

		else:
			return -1

	def print_state(self):
		print "\n\n\nScore :", self.score
		print "New State : "
		print_table(self.array, self.len)
		# print tabulate([self.array[0], self.array[1], self.array[2], self.array[3])
		print "\n\n\n\n\n\n\n\n\n\n\n"

class Agent(object):

	def __init__(self, env, extra_feats = False):
		self.env = env
		# 1 for won, 0 for in progress, -1 for lost
		self.game_status = 0
		self.handcrafted = extra_feats

	def take_step(self, step):
		prev_score = self.env.score
		# if (step == 0):
		# 	print "Up"
		# elif (step == 1):
		# 	print "Right"
		# elif (step == 2):
		# 	print "Down"
		# elif (step == 3):
		# 	print "Left"
		ret = self.env.take_step(step)
		# time.sleep(2)
		
		if (self.env.game_lost()):
			self.game_status = -1
		elif (self.env.game_won):
			self.game_status = 1

		return self.game_status, self.reward_formulation(prev_score)

	def reward_formulation(self, prev_score):
		reward = self.env.score - prev_score
		if (reward >= 512):
			reward = 0.5
		elif (reward >= 32):
			reward = 0.3
		elif (reward > 0):
			reward = 0.1
		if (self.game_status == 1):
			reward = 1.0
		elif (self.game_status == -1):
			reward = -1.0
		return reward

	def get_array(self):
		# if (self.handcrafted):
		# 	extra_features = np.zeros([4])
		# 	extra_features[0] = self.env.unfilled
		# 	extra_features[1] = self.env.max
		# 	extra_features[2] = np.unique(self.env.array).size

		# 	for x in xrange(self.env.len):
		# 		for y in xrange(self.env.len-1):
		# 			if (self.env.array[x][y] == self.env.array[x][y+1] and self.env.array[x][y] != 0):
		# 				extra_features[3] += 1
		# 			if (self.env.array[y][x] == self.env.array[y+1][x] and self.env.array[y][x] != 0):
		# 				extra_features[3] += 1
		# 	return np.append(np.array(self.env.array), extra_features).reshape([1,self.env.size + 4])
		# else :
		# 	return np.array(self.env.array).reshape([1,self.env.size])

		ret_arr = np.copy(self.env.array)
		ret_arr[ret_arr == 0] = 1
		return np.eye(12)[np.log2(ret_arr).astype(int)]


if __name__ == '__main__':

	game = Env(True,3);
	agent = Agent(game);

	while(1):
		c = raw_input()
		if (c == 'w'):
			print agent.take_step(0)
		elif (c == 'd'):
			print agent.take_step(1)
		elif (c == 's'):
			print agent.take_step(2)
		elif (c == 'a'):
			print agent.take_step(3)


	# while(1):
	# 	c = random.randrange(4)
	# 	print
	# 	agent.take_step(c)

