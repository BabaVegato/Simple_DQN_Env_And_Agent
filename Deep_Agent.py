from __future__ import print_function, division
from builtins import range

import numpy as np
from matplotlib.pyplot import *
from graphics import *
import random
from Terrain import *
import time
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
	def __init__(self, ALPHA):
		super(DeepQNetwork, self).__init__()
		self.fc1 = nn.Linear(in_features = 25, out_features = 128)
		self.fc2 = nn.Linear(in_features = 128, out_features = 4)

		self.optimizer = optim.RMSprop(self.parameters(), lr = ALPHA)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		t = T.tensor(state)
		t = F.relu(self.fc1(t))
		t = self.fc2(t)
		return t

class Agent:
	def __init__(self, batch_size_learn, gamma, epsilon, alpha, maxMemorySize, epsEnd=0.05, replace=10000):
		self.batch_size_learn = batch_size_learn
		self.GAMMA = gamma
		self.EPS_END = epsEnd
		self.steps = 0
		self.memSize = 500
		self.learn_step_counter = 0
		self.memory = []
		self.memCount = 0
		self.replace_target_count = 10000
		self.Q_eval = DeepQNetwork(alpha)
		self.Q_next = DeepQNetwork(alpha)

	def storeTransition(self, state, action, reward, state_):
		if self.memCount < self.memSize:
			self.memory.append([state, action, reward, state_])
		else:
			self.memory[self.memCount % self.memSize] = [state, action, reward, state_]
		self.memCount+=1

	def decide_Action(self, state, eps):
		actions = self.Q_eval.forward(state)
		if np.random.random() < eps:
			return random.randint(0, 3)
		else:
			return T.argmax(actions[1]).item()

	def learn(self):
		#Zero the gradients for batch optimization
		self.Q_eval.optimizer.zero_grad()
		#check if it's time to replace the target network : target network <- Qeval
		if self.replace_target_count is not None and self.learn_step_counter % self.replace_target_count == 0:
			self.Q_next.load_state_dict(self.Q_eval.state_dict())

		#Create a memory by making a batch
		if self.memCount + self.batch_size_learn < self.memSize:
			memStart = int(np.random.choice(range(self.memCount)))
		else:
			memStart = int(np.random.choice(range(self.memCount - self.batch_size_learn-1)))
		miniBatch = self.memory[memStart:memStart + self.batch_size_learn]
		memory = np.array(miniBatch)

		#feedforward the current state & the successor state
		Qpred = self.Q_eval.forward(list(memory[:, 0])).to(self.Q_eval.device)
		Qnext = self.Q_next.forward(list(memory[:, 3])).to(self.Q_eval.device)
		maxAction = T.argmax(Qnext)
		print(maxAction)
		rewards = T.Tensor(list(memory[:,2]))
		Qtarget = Qpred
		Qtarget[maxAction] = rewards + self.GAMMA*Qnext[T.argmax(Qnext)]

		if self.steps > 500:
			if self.EPSILON - 1e-4 > self.EPS_END:
				self.EPSILON -= 1e-4
			else: 
				self.EPSILON = self.EPS_END
		loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
		loss.backward()
		self.Q_eval.optimizer.step()
		self.learn_step_counter += 1

def Get_State(obs):
	state = [0 for i in range(25)]
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

def Play_One_Game(agent, eps, gamma, iters_max, learn, watch):
	iters = 0
	total_reward = 0
	win = 0

	env, player, obs = StartEnv(15)

	while env.done is not True and iters_max > iters :
		state = Get_State(obs)

		action = agent.decide_Action(state, eps)
		obs, reward = Action_On_Env(env, player, action)
		state_ = Get_State(obs)

		agent.storeTransition(state, action, reward, state_)
		state = state_

		if watch :
			if iters == 0 :
				initDisp = True
			else :
				initDisp = False

			win = Display_World(env, win, obs, initDisp)

		if learn :
			agent.learn()
		

		iters+=1

	return reward

def initMemory(agent):

	while agent.memCount < agent.memSize :
		env, player, obs = StartEnv(15)

		while not env.done :
			state = Get_State(obs)

			action = random.randint(0, len(env.action_space))
			obs, reward = Action_On_Env(env, player, action)
			state_ = Get_State(obs)

			agent.storeTransition(state, action, reward, state_)
			state = state_
	print("Done initializing memory.")
	

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
	agent = Agent(gamma=0.95, epsilon=1.0, alpha=0.003, maxMemorySize=5000, replace=None)
	total_reward = Play_One_Game(env, player, agent, first_obs, 
		0, gamma, iters_max, False, True)
	
def train(nbGames, batch_size):
	start_time = time.time()

	agent = Agent(gamma=0.95, batch_size_learn=32, epsilon=1.0, alpha=0.003, maxMemorySize=10000, replace=None)

	print("Initializing the agent's memory ...")
	initMemory(agent)

	print("Start training...")

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
		
		total_reward = Play_One_Game(agent, eps, gamma, iters_max, True, False)
		list_rewards.append(total_reward)
		if i==nb_games-1:
			print("--- %s seconds ---" % (time.time() - start_time))

	plot_mean(list_rewards, 50)

if __name__ == '__main__':
	
	nb_games = 5000
	batch_size = 32

	#just_watch_a_game(True)
	train(nb_games, batch_size)