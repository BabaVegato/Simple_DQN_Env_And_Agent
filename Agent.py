from __future__ import print_function, division
from builtins import range

import numpy as np
from matplotlib.pyplot import *
from graphics import *
import random
from Terrain_facile import *
import time

class Agent:
	def __init__(self, env, ENV_SIZE, Q):
		self.env = env
		self.length_of_obs = (env.SIZE_OBS*3)**2

		# Number of elements inside obs * number of different blocks
		num_states = env.number_of_different_blocks ** self.length_of_obs
		num_actions = env.num_actions
		if Q is 0:
			self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
		else : 
			self.Q = Q

	def decide_Action(self, state, eps):
		if np.random.random() < eps:
			return random.randint(0, len(self.Q[state])-1)
		else:
			return np.argmax(self.Q[state])

	def update_Q(self, state, action, G):
		self.Q[state, action] += 1e-2*(G - self.Q[state, action])


def Get_State(obs):
	state = 0
	ind = 0
	for i in range(len(obs)) :
		for j in range(len(obs)) :
			if(type(obs[i][j]) is Tree):
				state += 0
			if(type(obs[i][j]) is Ground):
				state += 1*(3**ind)
			if(type(obs[i][j]) is Hole):
				state += 2*(3**ind)
			ind+=1
	return state

def Play_One_Game(env, player, agent, obs, eps, gamma, iters_max, learn, watch):
	iters = 0
	total_reward = 0
	win = 0

	while env.done is not True and iters_max > iters :
		prev_state = Get_State(obs)

		action = agent.decide_Action(prev_state, eps)
		obs, reward = Action_On_Env(env, player, action)

		if watch :
			if iters == 0 :
				initDisp = True
			else :
				initDisp = False

			win = Display_World(env, win, obs, initDisp)

		if learn :
			G = reward + gamma*np.max(agent.Q[Get_State(obs)])
			agent.update_Q(prev_state, action, G)
		

		iters+=1

	return reward, agent.Q

def plot_mean(list, scale):
	N = len(list)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.asarray(list[max(0, t-scale):(t+1)]).mean()
	plot(running_avg)
	xlabel('Game number')
	ylabel('Total rewards averaged')
	title('Training results, over ' + str(len(list)) + ' games played')
	show()

def just_watch_a_game(already_Trained):
	SIZE = 30
	iters_max = 1000
	gamma = 0.9
	env, player, first_obs = StartEnv(SIZE)
	if already_Trained:
		Q = np.load('Qmatrix.npy')
	else :
		Q = 0
	agent = Agent(env, SIZE, Q)
	total_reward, Q = Play_One_Game(env, player, agent, first_obs, 
		0, gamma, iters_max, False, True)
	
def train(nb_games):
	start_time = time.time()
	print("Start training...")
	#Uncomment if you want to capitalize on your training
	Q = np.load('Qmatrix.npy')
	#Q = 0

	iters_max = 1000
	gamma = 0.9
	SIZE = 30
	
	list_rewards = []

	for i in range(nb_games): 
		eps = 1/(i+1)
		if i == 0 :
			print('Starting')
		if i%100 == 0:
			print('Game ', i)
		env, player, first_obs = StartEnv(SIZE)
		agent = Agent(env, SIZE, Q)
		total_reward, Q = Play_One_Game(env, player, agent, first_obs, 
			eps, gamma, iters_max, True, False)
		list_rewards.append(total_reward)
		if i==nb_games-1:
			np.save( 'Qmatrix.npy' , Q)
			print("Qmatrix saved.")
			print("--- %s seconds ---" % (time.time() - start_time))

	plot_mean(list_rewards, 50)

if __name__ == '__main__':
	
	nb_games = 500
	already_Trained = True
	#just_watch_a_game(already_Trained)
	train(nb_games)