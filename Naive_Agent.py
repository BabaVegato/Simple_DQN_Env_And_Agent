from __future__ import print_function, division
from builtins import range
from collections import deque
import sys

import numpy as np
from matplotlib.pyplot import *
from graphics import *
import random
from Terrain_facile import *
import time
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
	def __init__(self, input_Size, lr):
		super(DeepQNetwork, self).__init__()
		self.fc1 = nn.Linear(in_features = input_Size, out_features = 128)
		self.fc2 = nn.Linear(in_features = 128, out_features = 128)
		self.fc3 = nn.Linear(in_features = 128, out_features = 4)

		self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		t = T.tensor(state)
		t = F.relu(self.fc1(t))
		t = F.relu(self.fc2(t))
		t = self.fc3(t)
		return t

class Agent:
	def __init__(self, size_Obs, gamma, epsilon, lr):
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_dec = 1e-5
		self.eps_min = 1e-2
		self.input_Size = (size_Obs*2 + 1)**2
		self.Network = DeepQNetwork(self.input_Size, lr)

	def decide_Action(self, state, see_details):
		
		if np.random.random() < self.epsilon:
			return random.randint(0, 3)
		else:
			actions = self.Network.forward(state)
			if see_details:
				print("Decides what's best : ")
				print(actions)
				print("action chosen", T.argmax(actions).item())
			return T.argmax(actions).item()
	
	def save(self, size_Obs):
		state = {
		'eps' : self.epsilon,
    	'state_dict': self.Network.state_dict(),
    	'optimizer': self.Network.optimizer.state_dict()
		}
		T.save(state, 'model' + str(size_Obs) + '.pth')
		print('Model saved !')

	def load(self, size_Obs):
		state = T.load('model' + str(size_Obs) + '.pth')
		self.Network.load_state_dict(state['state_dict'])
		self.Network.optimizer.load_state_dict(state['optimizer'])
		self.epsilon = state['eps']
		print('Model loaded !')

	def decrement_epsilon(self):
		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

	def learn(self, state, action, reward, state_):
		self.Network.optimizer.zero_grad()
		states = T.tensor(state, dtype = T.float).to(self.Network.device)
		actions = T.tensor(action).to(self.Network.device)
		rewards = T.tensor(reward).to(self.Network.device)
		states_ = T.tensor(state_, dtype = T.float).to(self.Network.device)

		q_pred = self.Network.forward(states)[actions]
		q_next = self.Network.forward(states_).max()
		q_target = reward + self.gamma*q_next

		loss = self.Network.loss(q_target, q_pred).to(self.Network.device)
		loss.backward()
		self.Network.optimizer.step()
		self.decrement_epsilon()

def Get_State(obs):
	state = [0 for i in range(len(obs)**2)]
	ind = 0
	for rangeeObs in obs :
		for elementObs in rangeeObs :
			if(type(elementObs) is Tree):
				state[ind] = 0.
			if(type(elementObs) is Ground):
				state[ind] = 1.
			if(type(elementObs) is Hole):
				state[ind] = 2.
			if(type(elementObs) is Player):
				state[ind] = 3.
			ind+=1
	return state

def Play_One_Game(size_Obs, agent, gamma, iters_max, learn, watch):
	iters = 0
	total_reward = 0
	win = 0
	SIZE = 40

	env, player, obs = StartEnv(SIZE, size_Obs)
	while env.done is not True and iters_max > iters :
		state = Get_State(obs)

		action = agent.decide_Action(state, watch)

		obs_, reward = Action_On_Env(env, player, action)
		state_ = Get_State(obs_)

		if watch :
			if iters == 0 :
				initDisp = True
			else :
				initDisp = False

			win = Display_World(env, win, obs_, initDisp)

		if learn :
			agent.learn(state, action, reward, state_)
		obs = obs_
		iters+=1
		#time.sleep(5)
	return reward
	
def plot_mean(listedeque, scale):
	liste = list(listedeque)
	N = len(liste)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.asarray(liste[max(0, t-scale):(t+1)]).mean()
	plot(running_avg)
	xlabel('Game number')
	ylabel('Total rewards averaged')
	title('Training results, over ' + str(len(liste)) + ' games played')
	show()

def just_watch_a_game(size_Obs):
	agent = Agent(size_Obs, gamma=0.95, epsilon=0, lr=0.0001)
	agent.load(size_Obs)
	agent.Network.eval()

	print("Start the game...")

	iters_max = 350
	g = 0.9
	SIZE = 30

	#Play
	total_reward = Play_One_Game(size_Obs=size_Obs, agent = agent, gamma = g, iters_max = iters_max, learn = False, watch = True)
	
def train(nbGames, size_Obs, new):
	start_time_Total = time.time()

	agent = Agent(size_Obs, gamma=0.95, epsilon=1.0, lr=0.0001)

	if not new :
		agent.load(size_Obs)

	print("Start training...")

	iters_max = 500
	g = 0.9
	SIZE = 30
	
	list_rewards = deque([])

	for i in range(nb_games): 
		if i%100 == 0 and i > 51:
			print('Game ', i)
			avg_rew = 0
			for j in range(50):
				avg_rew += list_rewards[-j]
			avg_rew = avg_rew/50
			print("Average reward : ", int(avg_rew))
		#Play
		total_reward = Play_One_Game(size_Obs = size_Obs, agent = agent, gamma = g, iters_max = iters_max, learn = True, watch = False)
		list_rewards.append(total_reward)
		if i==nb_games-1:
			if nb_games>=1000:
				agent.save(size_Obs)
			print("--- %s seconds ---" % (time.time() - start_time_Total))
	
	plot_mean(list_rewards, int(nb_games/50))

if __name__ == '__main__':
	
	args = sys.argv
	new = False

	size_Obs = 2

	if len(args) == 1:
		nb_games = 3000
		train(nb_games, new)

	else :
		if 'train' in args[1]:
			nb_games = int(args[2])
			if len(args)>3:
				if 'new' in args[3]:
					new = True
			train(nb_games, size_Obs, new)
		if 'play' in args[1] :
			just_watch_a_game(size_Obs)
	