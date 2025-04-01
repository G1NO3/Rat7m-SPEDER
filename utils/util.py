import time
import gym 
import numpy as np 
import torch

from torch import nn
from torch.nn import functional as F


def unpack_batch(batch):
  return batch.state, batch.action, batch.next_state, batch.reward, batch.done, batch.task, batch.next_task


class Timer:

	def __init__(self):
		self._start_time = time.time()
		self._step_time = time.time()
		self._step = 0

	def reset(self):
		self._start_time = time.time()
		self._step_time = time.time()
		self._step = 0

	def set_step(self, step):
		self._step = step
		self._step_time = time.time()

	def time_cost(self):
		return time.time() - self._start_time

	def steps_per_sec(self, step):
		sps = (step - self._step) / (time.time() - self._step_time)
		self._step = step
		self._step_time = time.time()
		return sps


def eval_policy(policy, eval_env, eval_episodes=10):
	"""
	Eval a policy
	"""
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward



def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)


class MLP(nn.Module):
	def __init__(self,
								input_dim,
								hidden_dim,
								output_dim,
								hidden_depth=1,
								output_mod=None):
		super().__init__()
		self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
											output_mod)
		self.apply(weight_init)

	def forward(self, x):
		return self.trunk(x)

class DoubleMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 ):
        super().__init__()
        self.trunk1 = mlp_nobatchnorm(input_dim, hidden_dim, output_dim, hidden_depth)
        self.trunk2 = mlp_nobatchnorm(input_dim, hidden_dim, output_dim, hidden_depth)

    def forward(self, x):
        return self.trunk1(x), self.trunk2(x)



def mlp_nobatchnorm(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
	if hidden_depth == 0:
		mods = [nn.Linear(input_dim, output_dim)]
	else:
		mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
		for i in range(hidden_depth - 1):
			mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
		mods.append(nn.Linear(hidden_dim, output_dim))
	if output_mod is not None:
		mods.append(output_mod)
	trunk = nn.Sequential(*mods)
	return trunk

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
	if hidden_depth == 0:
		mods = [nn.Linear(input_dim, output_dim)]
	else:
		mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
		for i in range(hidden_depth - 1):
			mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
		mods.append(nn.Linear(hidden_dim, output_dim))
	if output_mod is not None:
		mods.append(output_mod)
	trunk = nn.Sequential(*mods)
	return trunk


# class truncated_normal(pyd.transformed_distribution.TransformedDistribution):
#     def __init__(self, mean, std, low, high):
#         self.mean = mean
#         self.std = std
#         self.low = low
#         self.high = high
#         self.normal_dist = Normal(mean, std)
#         # Transform the standard normal into a truncated range
#         # SigmoidTransform maps (-inf, inf) -> (0, 1)
#         # AffineTransform scales (0, 1) -> (low, high)
#         self.trunc_transform = torch.distributions.transforms.ComposeTransform([
#             SigmoidTransform(),  # Maps to (0, 1)
#             AffineTransform(loc=low, scale=high - low)  # Maps (0, 1) -> (low, high)
#         ])
#         super().__init__(self.normal_dist, self.trunc_transform)

class SigmoidMLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.l1 = nn.Linear(input_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, output_dim)
	def forward(self, x):
		x = F.elu(self.l1(x))
		x = nn.Sigmoid()(self.l2(x))
		return x

class Norm1MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.l1 = nn.Linear(input_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, output_dim)
	def forward(self, x):
		x = F.elu(self.l1(x))
		x = F.elu(self.l2(x))
		x = x / torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
		return x

class Norm1MLP_singlelayer(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.l1 = nn.Linear(input_dim, output_dim)
	def forward(self, x):
		x = F.elu(self.l1(x))
		x = x / torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
		return x

class RFFMLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.l0 = nn.Linear(input_dim, hidden_dim) #TODO:区分Phi和Mu并分别用[I I]和I初始化
		# if input_dim == (2*hidden_dim):
		# 	self.l0.weight.data = torch.cat([torch.eye(hidden_dim), torch.eye(hidden_dim)], dim=1)
		# 	nn.init.constant_(self.l0.bias, 0)
		# elif input_dim == hidden_dim:
		# 	nn.init.eye_(self.l0.weight)
		# 	nn.init.constant_(self.l0.bias,0)
		# else:
		# 	print(input_dim==2*hidden_dim)
		# 	print(input_dim-2*hidden_dim)
		# 	print('input:',input_dim,type(input_dim),'hidden_dim',hidden_dim,type(hidden_dim))
		# 	raise NotImplementedError
		nn.init.normal_(self.l0.weight, mean=0, std=1)
		nn.init.normal_(self.l0.bias, mean=0, std=1)
		self.l1 = nn.Linear(hidden_dim, output_dim)  # random feature
		nn.init.normal_(self.l1.weight, mean=0, std=1)
		nn.init.normal_(self.l1.bias, mean=0, std=1)
		print('Initialize')
		print(self.l1.weight.shape, self.l1.weight.mean(), self.l1.weight.std())
		print(self.l1.bias.shape, self.l1.bias.mean(), self.l1.bias.std())
		self.l1.weight.requires_grad_(False)
		self.l1.bias.requires_grad_(False)
	def forward(self, phi_feed_feature):
		z1 = self.l0(phi_feed_feature)
		z2 = torch.cos(self.l1(z1))
		return z2
	
class RFFMLP_notrain(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.l1 = nn.Linear(input_dim, output_dim)  # random feature
		nn.init.normal_(self.l1.weight, mean=0, std=5)
		nn.init.normal_(self.l1.bias, mean=0, std=5)
		print('Initialize')
		print(self.l1.weight.shape, self.l1.weight.mean(), self.l1.weight.std())
		print(self.l1.bias.shape, self.l1.bias.mean(), self.l1.bias.std())
		self.l1.weight.requires_grad_(False)
		self.l1.bias.requires_grad_(False)
	def forward(self, phi_feed_feature):
		z2 = torch.cos(self.l1(phi_feed_feature))
		return z2

class RFF_complex_critic(nn.Module):
	def __init__(self, feature_dim, hidden_dim):
		super().__init__()
		self.l1 = nn.Linear(feature_dim, hidden_dim)  # random feature
		self.l2_1 = nn.Linear(hidden_dim, hidden_dim)
		self.l3_1 = nn.Linear(hidden_dim, 1)
		self.l2_2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3_2 = nn.Linear(hidden_dim, 1)
	def forward(self, critic_feed_feature):
		z1 = torch.cos(self.l1(critic_feed_feature))
		z2_1 = F.elu(self.l2_1(z1))
		z3_1 = self.l3_1(z2_1)
		z2_2 = F.elu(self.l2_2(z1))
		z3_2 = self.l3_2(z2_2)
		q = z3_1 * z3_2
		return q

class RFFCritic(nn.Module):

	def __init__(self, feature_dim, hidden_dim):
		super().__init__()

        # Q1
		self.l1 = nn.Linear(feature_dim, hidden_dim)  # random feature
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

        # Q2
		self.l4 = nn.Linear(feature_dim, hidden_dim)  # random feature
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)

		self.outputs = dict()

	def forward(self, critic_feed_feature):
		q1 = torch.sin(self.l1(critic_feed_feature))
		q1 = F.elu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = torch.sin(self.l4(critic_feed_feature))
		q2 = F.elu(self.l5(q2))
		q2 = self.l6(q2)

		self.outputs['q1'] = q1
		self.outputs['q2'] = q2

		return q1, q2


class Theta(nn.Module):
    """
    Linear theta
    <phi(s, a), theta> = r
    """

    def __init__(
            self,
            feature_dim=1024,
    ):
        super(Theta, self).__init__()
        self.l = nn.Linear(feature_dim, 1)

    def forward(self, feature):
        r = self.l(feature)
        return r

def to_np(t):
	if t is None:
		return None
	elif t.nelement() == 0:
		return np.array([])
	else:
		return t.cpu().detach().numpy()