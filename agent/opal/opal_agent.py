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
from agent.sac.actor import DiagGaussianActor, MultiSoftmaxActor, AutoregressiveGaussianActor, DiagGaussianEncoder
from torchinfo import summary
import numpy as np
from functools import partial
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OpalAgent(SACAgent):
    def __init__(
            self,
            state_dim,
            action_dim,
            lr=1e-3,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=256,
            feature_tau=0.005,
            feature_dim=2048,  # latent feature dim
            use_feature_target=True,
            extra_feature_steps=1,
            device='cuda:0',
            state_dataset=None,
            lasso_coef=1e-3,
            n_task=3,
            beta=0.1,
            actor_type='gaussian',
            directory=None,
            **kwargs
    ):

        # state_dataset = state_task_dataset[:, :state_dim]
        # mean, std = state_dataset.mean(0), state_dataset.std(0)
        # low, high = state_dataset.min(0)[0], state_dataset.max(0)
        # self.low, self.high = low, high
        # self.obs_dist = pyd.Uniform(low=torch.FloatTensor(low).to(device), high=torch.FloatTensor(high).to(device))

        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.n_action = 5
        self.hidden_dim = hidden_dim    
        self.use_feature_target = use_feature_target
        self.extra_feature_steps = extra_feature_steps
        self.discount = discount
        self.device = device
        self.log_alpha = torch.tensor(np.log(alpha)).to(self.device)
        self.log_alpha.requires_grad = True
        self.steps = 0
        self.beta = beta
        self.n_task = n_task
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)).to(self.device)
        self.rnn = nn.GRU(input_size=hidden_dim+action_dim,
                              hidden_size=hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True,
                              ).to(self.device)
        self.latent_encoder = DiagGaussianEncoder(obs_dim=2*hidden_dim, action_dim=hidden_dim,
                                                 hidden_dim=hidden_dim, hidden_depth=0,
                                                 
			log_std_bounds=[-5., 2.], ).to(self.device)
        self.prior_encoder = DiagGaussianEncoder(obs_dim=state_dim, action_dim=hidden_dim,
                                                    hidden_dim=hidden_dim, hidden_depth=1,
			log_std_bounds=[-5., 2.], ).to(self.device)
        self.actor = DiagGaussianEncoder(obs_dim=hidden_dim+state_dim, action_dim=action_dim, 
                                        hidden_dim=hidden_dim, hidden_depth=2,
			log_std_bounds=[-5., 2.], ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.state_processor.parameters()) +
            list(self.rnn.parameters()) +
            list(self.latent_encoder.parameters()) +
            list(self.prior_encoder.parameters()) +
            list(self.actor.parameters()),
            lr=lr
        )
    def train(self, buffer, batch_size, seq_len):
        """
        One train step
        """
        self.steps += 1

        # Feature step
        # for _ in range(self.extra_feature_steps + 1):
        batch_1 = buffer.sample_sequence(batch_size, seq_len)
        info = self.update(batch_1)
        return info

    def update(self, batch):
        expert_state, expert_action, expert_next_state, expert_reward, expert_done, expert_task, expert_next_task = unpack_batch(batch)
        B, T = expert_state.shape[0], expert_state.shape[1]
        assert expert_state.shape == (B, T, self.state_dim)
        assert expert_action.shape == (B, T, self.action_dim)
        assert expert_next_state.shape == (B, T, self.state_dim)
        processed_state = self.state_processor(expert_state.reshape(B*T, self.state_dim)).reshape(B, T, -1)
        rnn_input = torch.concat([processed_state, expert_action], dim=-1)
        rnn_latent, _ = self.rnn(rnn_input)
        assert rnn_latent.shape == (B, T, 2*self.hidden_dim)
        latent_dist = self.latent_encoder(rnn_latent.mean(dim=1))
        z = latent_dist.rsample()
        assert z.shape == (B, self.hidden_dim)
        starting_state = expert_state[:, 0, :]
        prior_dist = self.prior_encoder(starting_state)
        kl_loss = torch.mean(pyd.kl_divergence(latent_dist, prior_dist))
        z_repeat = z.reshape(B, 1, self.hidden_dim).repeat(1, T, 1)
        actor_input = torch.concat([z_repeat, expert_state], dim=-1).reshape(B*T, self.hidden_dim + self.state_dim)
        actor_dist = self.actor(actor_input)
        assert actor_dist.mean.shape == (B*T, self.action_dim)
        action = expert_action.reshape(B*T, self.action_dim)
        log_prob = actor_dist.log_prob(action).reshape(B, T, self.action_dim).sum(-1)
        actor_loss = -log_prob.mean(-1).mean()

        loss = actor_loss + self.beta * kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def action_loglikelihood(self, state, action, cut_seq_len=10):
        B = state.shape[0] // cut_seq_len
        T = cut_seq_len
        state_1 = state.reshape(B, T, self.state_dim)
        action_1 = action.reshape(B, T, self.action_dim)
        processed_state = self.state_processor(state_1.reshape(B*T, self.state_dim)).reshape(B, T, -1)
        rnn_input = torch.concat([processed_state, action_1], dim=-1)
        rnn_latent, _ = self.rnn(rnn_input)
        assert rnn_latent.shape == (B, T, 2*self.hidden_dim)
        latent_dist = self.latent_encoder(rnn_latent.mean(dim=1))
        z = latent_dist.rsample()
        assert z.shape == (B, self.hidden_dim)
        z_repeat = z.reshape(B, 1, self.hidden_dim).repeat(1, T, 1)
        actor_input = torch.concat([z_repeat, state_1], dim=-1).reshape(B*T, self.hidden_dim + self.state_dim)
        actor_dist = self.actor(actor_input)
        assert actor_dist.mean.shape == (B*T, self.action_dim)
        action = action_1.reshape(B*T, self.action_dim)
        log_prob = actor_dist.log_prob(action).reshape(B, T, self.action_dim).sum(-1).reshape(-1)
        return log_prob

    def state_dict(self):
        return {
            'state_processor': self.state_processor.state_dict(),
            'rnn': self.rnn.state_dict(),
            'latent_encoder': self.latent_encoder.state_dict(),
            'prior_encoder': self.prior_encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'log_alpha': self.log_alpha
        }
    
    def load_state_dict(self, state_dict):
        self.state_processor.load_state_dict(state_dict['state_processor'])
        self.rnn.load_state_dict(state_dict['rnn'])
        self.latent_encoder.load_state_dict(state_dict['latent_encoder'])
        self.prior_encoder.load_state_dict(state_dict['prior_encoder'])
        self.actor.load_state_dict(state_dict['actor'])
        self.log_alpha = state_dict['log_alpha']
    
    def n_param(self):
        modules = [self.rnn, self.latent_encoder, self.prior_encoder, self.actor]
        n = sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)
        return n