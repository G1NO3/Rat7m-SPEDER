import collections
import numpy as np
import torch



Batch = collections.namedtuple(
	'Batch',
	['state', 'action', 'next_state', 'reward', 'done', 'task', 'next_task']
	)

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda:0'):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.task = np.zeros((max_size, 1))
		self.next_task = np.zeros((max_size, 1))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, next_state, reward, done, task, next_task):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done
		self.task[self.ptr] = task
		self.next_task[self.ptr] = next_task

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return Batch(
			state=torch.FloatTensor(self.state[ind]).to(self.device),
			action=torch.FloatTensor(self.action[ind]).to(self.device),
			next_state=torch.FloatTensor(self.next_state[ind]).to(self.device),
			reward=torch.FloatTensor(self.reward[ind]).to(self.device),
			done=torch.FloatTensor(self.done[ind]).to(self.device),
			task=torch.FloatTensor(self.task[ind]).to(self.device),
			next_task=torch.FloatTensor(self.next_task[ind]).to(self.device)
		)

	def state_dict(self):
		return {
			'state': self.state[:self.size],
			'action': self.action[:self.size],
			'next_state': self.next_state[:self.size],
			'reward': self.reward[:self.size],
			'done': self.done[:self.size],
			'task': self.task[:self.size],
			'next_task': self.next_task[:self.size],
			'ptr': self.ptr,
			'size': self.size
		}
	def load_state_dict(self, state_dict):
		self.state = state_dict['state']
		self.action = state_dict['action']
		self.next_state = state_dict['next_state']
		self.reward = state_dict['reward']
		if 'task' in state_dict:
			self.task = state_dict['task']
			self.next_task = state_dict['next_task']
		self.done = state_dict['done']
		self.ptr = state_dict['ptr']
		self.size = state_dict['size']



