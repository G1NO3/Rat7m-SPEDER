import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal, SigmoidTransform, AffineTransform, TransformedDistribution
from torch import distributions as pyd
import os

# from utils.util import unpack_batch, RunningMeanStd
from utils.util import unpack_batch
from utils.util import MLP, DoubleMLP, RFFCritic, Theta, \
    RFFMLP, RFF_complex_critic, RFFMLP_notrain, Norm1MLP, Norm1MLP_singlelayer, \
    SigmoidMLP

from agent.sac.sac_agent import SACAgent, DoubleQCritic
from agent.sac.actor import DiagGaussianActor, MultiSoftmaxActor, AutoregressiveGaussianActor
from torchinfo import summary
import numpy as np
from functools import partial
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class IQLAgent():
    def __init__(
            self,
            state_dim,
            action_dim,
            discount=0.99,
            target_update_period=2,
            hidden_dim=1024,
            device='cuda:0',
            n_task=3,
            learnable_temperature=False,
            lr=1e-3,
            directory=None,
            **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.n_task = n_task
        self.device = device
        self.critic = DoubleMLP(input_dim=self.state_dim + self.action_dim, # try single task
                          output_dim=1,
                          hidden_dim=hidden_dim,
                            hidden_depth=1).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.tau = 0.005
        self.steps = 0
        self.discount = discount
        self.target_update_period = target_update_period
        self.learnable_temperature = learnable_temperature
        self.target_entropy = -action_dim
        self.actor = DiagGaussianActor(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=2,
            log_std_bounds=[-5., 2.],
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=lr)
        self.log_alpha = torch.tensor(np.log(1.0)).to(self.device)
        # self.log_alpha.requires_grad = True
        # self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        print('n_task:', n_task)
        self.n_task = n_task
        self.task_all = torch.eye(n_task).to(device)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def update_target(self):
        if self.steps % self.target_update_period == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def getQ(self, state, action):
        return self.critic(torch.concat([state, action], -1))
    def getV(self, state):
        dist = self.actor(state)
        action = dist.rsample().detach()
        q1, q2 = self.getQ(state, action)
        q = torch.min(q1, q2)
        v = q - self.alpha.detach() * dist.log_prob(action).sum(-1, keepdim=True)
        return v
    def get_targetQ(self, state, action):
        return self.critic_target(torch.concat([state, action], -1))
    def get_targetV(self, state):
        dist = self.actor(state)
        action = dist.sample()
        target_q1, target_q2 = self.get_targetQ(state, action)
        target_q = torch.min(target_q1, target_q2)
        target_v = target_q - self.alpha.detach() * dist.log_prob(action).sum(-1, keepdim=True)
        return target_v
    def iq_loss(self, current_Q, current_v, next_v, done):
        iq_alpha = 0.5
        y = (1 - done) * self.discount * next_v
        r = current_Q - y
        loss_1 = -r.mean()
        loss_2 = (current_v - y).mean()
        loss_3 = 1/(4*iq_alpha) * (r**2).mean()
        iql_loss = loss_1 + loss_2 + loss_3
        return iql_loss
    def critic_step(self, batch):
        state, action, next_state, reward, done, task, next_task = unpack_batch(batch)
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim
        assert done.shape[-1] == 1
        # task_onehot = self.task_all[task.long().reshape(-1)].to(self.device)
        # next_task_onehot = self.task_all[next_task.long().reshape(-1)].to(self.device)
        current_q1, current_q2 = self.getQ(state, action)
        next_v = self.get_targetV(next_state).detach()
        current_v = self.getV(state)
        q1_iqlloss = self.iq_loss(current_q1, current_v, next_v, done)
        q2_iqlloss = self.iq_loss(current_q2, current_v, next_v, done)
        critic_loss = (q1_iqlloss + q2_iqlloss) / 2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return {
            'q1_iqlloss': q1_iqlloss.item(),
            'q2_iqlloss': q2_iqlloss.item(),
            'critic_loss': critic_loss.item()
        }
    def update_actor_and_alpha(self, batch):
        state, action, next_state, reward, done, task, next_task = unpack_batch(batch)
        # task_onehot = self.task_all[task.long().reshape(-1)].to(self.device)
        dist = self.actor(state)
        # dist = self.actor(state)
        sample_action = dist.rsample()
        sample_q1, sample_q2 = self.getQ(state, sample_action)
        sample_q = torch.min(sample_q1, sample_q2)
        sample_action_logprob = dist.log_prob(sample_action).sum(-1, keepdim=True)
        SAC_loss = (self.alpha * sample_action_logprob - sample_q).mean()
        actor_loss = SAC_loss
        ###Behavior Cloning
        # log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # actor_loss = -log_prob.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        info = {'actor_loss': actor_loss.item()}

        # if self.learnable_temperature:
        #     self.log_alpha_optimizer.zero_grad()
        #     alpha_loss = (self.alpha * (-sample_action_logprob - self.target_entropy).detach()).mean()
        #     alpha_loss.backward()
        #     self.log_alpha_optimizer.step()

        #     info['alpha_loss'] = alpha_loss
        #     info['alpha'] = self.alpha
        return info
    def train(self, buffer, batch_size):
        self.steps += 1
        critic_info = self.critic_step(buffer.sample(batch_size))
        actor_info = self.update_actor_and_alpha(buffer.sample(batch_size))
        self.update_target()
        return {
            **critic_info,
            **actor_info
        }
    def state_dict(self):
        return {'critic': self.critic.state_dict(),
                'log_alpha': self.log_alpha,
                'actor': self.actor.state_dict()}
    def load_state_dict(self, state_dict):
        self.critic.load_state_dict(state_dict['critic'])
        self.log_alpha = state_dict['log_alpha']
        self.actor.load_state_dict(state_dict['actor'])

    def action_loglikelihood(self, state, action):
        # self.actor.eval()
        # task_onehot = self.task_all[task.long().squeeze(1)].to(self.device)
        q1, q2 = self.getQ(state, action)
        q = torch.min(q1, q2)
        # actor_log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # return actor_log_prob.mean()
        return q