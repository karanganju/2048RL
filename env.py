import numpy as np
import random
import time
import prettytable

def print_table(array):
	t = prettytable.PrettyTable(header=False, hrules=prettytable.ALL, border=True)
	t.add_row(array[0])
	t.add_row(array[1])
	t.add_row(array[2])
	t.add_row(array[3])
	print t

class Env(object):

	def __init__(self, _is_human = False):
		# self.array = [[0,0,0,0],[8,4,4,0],[0,0,0,0],[0,0,0,0]]
		self.array = np.zeros((4,4), dtype=int)
		self.is_human = _is_human;
		self.score = 0;
		self.unfilled = 16;
		self.valid_move = False;
		self.has_merged = np.zeros(16)
		self.game_won = False
		self.max = 2
		# self.pp = pprint.PrettyPrinter(indent=10)
		self.take_step(debug_print = False);
		self.take_step();

	def game_lost(self):

		if self.unfilled > 0:
			return False

		for i in xrange(4):
			if (self.can_move(i)):
				return False

		return True


	def can_move(self, step):
		if (step == 0):
			for x in xrange(3):
				for y in xrange(4):
					if(self.array[x+1][y] != 0 and (self.array[x+1][y] == self.array[x][y] or self.array[x][y] == 0)):
						return True

		# Right
		elif (step == 1):
			for y in xrange(3):
				for x in xrange(4):
					if(self.array[x][2-y] != 0 and (self.array[x][2-y] == self.array[x][3-y] or self.array[x][3-y] == 0)):
						return True

		# Down
		elif (step == 2):
			for x in xrange(3):
				for y in xrange(4):
					if(self.array[2-x][y] != 0 and (self.array[2-x][y] == self.array[3-x][y] or self.array[3-x][y] == 0)):
						return True

		# Left
		elif (step == 3):
			for y in xrange(3):
				for x in xrange(4):
					if(self.array[x][y+1] != 0 and (self.array[x][y+1] == self.array[x][y] or self.array[x][y] == 0)):
						return True

		return False

	def take_step(self, step = -1, debug_print = True):

		debug_print = debug_print and self.is_human;
		self.valid_move = False;
		self.has_merged = np.zeros(16)

		# Might need to rewrite bottom part for readability

		# Up
		if (step == 0):
			for x in xrange(3):
				for y in xrange(4):
					for steps in xrange(x+1):
						if(self.move(x+1-steps,y,x-steps,y) == -1):
							break

		# Right
		elif (step == 1):
			for y in xrange(3):
				for x in xrange(4):
					for steps in xrange(y+1):
						if(self.move(x,2+steps-y,x,3+steps-y) == -1):
							break

		# Down
		elif (step == 2):
			for x in xrange(3):
				for y in xrange(4):
					for steps in xrange(x+1):
						if(self.move(2+steps-x,y,3+steps-x,y) == -1):
							break

		# Left
		elif (step == 3):
			for y in xrange(3):
				for x in xrange(4):
					for steps in xrange(y+1):
						if(self.move(x,y+1-steps,x,y-steps) == -1):
							break

		if (self.game_won):
			return 1

		if (self.valid_move or step == -1):
			filling_cell = random.randrange(self.unfilled);
			for x in xrange(16):
				if (self.array[x/4][x%4] == 0):
					if (filling_cell == 0):
						r = random.random()
						if (r < 0.9):
							self.array[x/4][x%4] = 2
						else:
							self.array[x/4][x%4] = 4
						self.unfilled-=1;
						break
					else:
						filling_cell-=1;

		if (debug_print):
			self.print_state()

		if (self.game_lost() == True):
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
		elif (self.has_merged[i2*4+j2] == 1):
			return -1

		elif (self.array[i2][j2] == 0 or self.array[i2][j2] == self.array[i][j]):
			ret_val = 0
			self.score += 2*self.array[i2][j2];
			if (self.array[i2][j2] != 0):
				self.unfilled+=1;
				ret_val = -1;
				self.has_merged[i2*4+j2] = 1
				self.max = max(self.max, self.array[i2][j2])
			self.array[i2][j2] = self.array[i][j] + self.array[i2][j2];
			self.array[i][j] = 0;
			if (self.array[i2][j2] == 2048):
				self.game_won = True
			self.valid_move = True;
			return ret_val

		else:
			return -1

	def print_state(self):
		print "\n\n\nScore :", self.score
		print "New State : "
		print_table(self.array)
		# print tabulate([self.array[0], self.array[1], self.array[2], self.array[3])
		print "\n\n\n\n\n\n\n\n\n\n\n"

class Agent(object):

	def __init__(self, env):
		self.env = env
		# 1 for won, 0 for in progress, -1 for lost
		self.game_status = 0

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
		
		if (env.game_lost()):
			self.game_status = -1
		elif (env.game_won()):
			self.game_status = 1

		return self.game_status,self.reward_formulation(prev_score)

	def reward_formulation(prev_score):
		if (self.game_status == 0):
			return self.env.score - prev_score
		elif (self.game_status == 1):
			return 1000
		elif (self.game_status == -1):
			return -1000

	def get_array(self):
		return self.env.array.reshape(16)



if __name__ == '__main__':

	game = Env(True);
	agent = Agent(game);

	while(1):
		c = raw_input()
		if (c == 'w'):
			game.take_step(0)
		elif (c == 'd'):
			game.take_step(1)
		elif (c == 's'):
			game.take_step(2)
		elif (c == 'a'):
			game.take_step(3)


	# while(1):
	# 	c = random.randrange(4)
	# 	print
	# 	agent.take_step(c)

