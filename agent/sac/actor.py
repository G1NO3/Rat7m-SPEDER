"""
We adapt the code from https://github.com/denisyarats/pytorch_sac
"""


import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions import Normal
from utils import util
from torch.distributions import Categorical

class TanhTransform(pyd.transforms.Transform):
  domain = pyd.constraints.real
  codomain = pyd.constraints.interval(-1.0, 1.0)
  bijective = True
  sign = +1

  def __init__(self, cache_size=1):
    super().__init__(cache_size=cache_size)

  @staticmethod
  def atanh(x):
    return 0.5 * (x.log1p() - (-x).log1p())

  def __eq__(self, other):
    return isinstance(other, TanhTransform)

  def _call(self, x):
    return x.tanh()

  def _inverse(self, y):
    # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
    # one should use `cache_size=1` instead
    eps = 1e-6
    y = y.clamp(-1 + eps, 1 - eps)
    return self.atanh(y)

  def log_abs_det_jacobian(self, x, y):
    # We use a formula that is more numerically stable, see details in the following link
    # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
    return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

    self.base_dist = pyd.Normal(loc, scale)
    transforms = [TanhTransform()]
    super().__init__(self.base_dist, transforms)

  @property
  def mean(self):
    mu = self.loc
    for tr in self.transforms:
        mu = tr(mu)
    return mu


class DiagGaussianActor(nn.Module):
  """torch.distributions implementation of an diagonal Gaussian policy."""
  def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                log_std_bounds):
    super().__init__()

    self.log_std_bounds = log_std_bounds
    self.trunk = util.mlp(obs_dim, hidden_dim, 2 * action_dim,
                            hidden_depth)

    self.outputs = dict()
    self.apply(util.weight_init)

  def forward(self, obs):
    mu, log_std = self.trunk(obs).chunk(2, dim=-1)

    # constrain log_std inside [log_std_min, log_std_max]
    log_std = torch.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                  1)

    std = log_std.exp()

    self.outputs['mu'] = mu
    self.outputs['std'] = std

    dist = pyd.Normal(mu, std)
    return dist
  def select_action(self, obs):
    with torch.no_grad():
      dist = self.forward(obs)
      action = dist.rsample()
      return action
  
class MultiSoftmaxActor(nn.Module):
  def __init__(self, obs_dim, action_dim, n_action, hidden_dim, hidden_depth):
    super().__init__()
    self.action_dim = action_dim
    self.n_action = n_action
    self.trunk = util.mlp(obs_dim, hidden_dim, action_dim,
                            hidden_depth)
    self.apply(util.weight_init)

  def forward(self, obs, temperature=1):
    logits = self.trunk(obs).reshape(*obs.shape[:-1], self.action_dim//self.n_action, self.n_action)
    dist = pyd.Categorical(logits=logits/temperature)
    return dist

  def select_action(self, obs):
    with torch.no_grad():
      dist = self.forward(obs)
      action = dist.sample()
      return action

class AutoregressiveGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_bounds = log_std_bounds  # tuple (log_std_min, log_std_max)
        
        # shared observation embed
        self.obs_embed = util.mlp(obs_dim, hidden_dim, hidden_dim, hidden_depth)
        # one small net per action-dim
        self.param_nets = nn.ModuleList([
            util.mlp(hidden_dim + i, hidden_dim, 2, hidden_depth)
            for i in range(action_dim)
        ])
        
        self.outputs = {}
        self.apply(util.weight_init)

    def forward(self, obs):
        """
        Sample actions and return their log-prob under the policy.
        Returns:
          actions:  [B x action_dim]
          log_prob: [B]
        """
        return self._sample_and_logprob(obs, rsample=True)

    def select_action(self, obs):
        with torch.no_grad():
            actions, _ = self._sample_and_logprob(obs, rsample=True)
            return actions

    def log_prob(self, obs, actions):
        """
        Compute the log-prob of given actions under the policy:
          obs:     [B x obs_dim]
          actions: [B x action_dim]
        Returns:
          log_prob: [B]
        """
        # reuse the same routine but without sampling
        _, log_prob = self._sample_and_logprob(obs, rsample=False, given_actions=actions)
        return log_prob

    def _sample_and_logprob(self, obs, rsample=True, given_actions=None):
        """
        Internal helper. If rsample=True, draws actions by .rsample().
        If given_actions is provided, uses those instead of sampling.
        Returns (actions, log_prob).
        """
        B = obs.shape[0]
        h = self.obs_embed(obs)  # [B x hidden_dim]
        
        actions   = []
        log_probs = []
        mus       = []
        stds      = []

        for i, net in enumerate(self.param_nets):
            # prepare input
            if i == 0:
                inp = h
            else:
                prev = torch.cat(actions, dim=-1)            # [B x i]
                inp  = torch.cat([h, prev], dim=-1)          # [B x (hidden_dim + i)]
            
            # get μ_i and raw log-σ_i
            mu_i, log_std_i = net(inp).chunk(2, dim=-1)      # each [B x 1]
            log_std_i = torch.tanh(log_std_i)
            lo, hi = self.log_std_bounds
            log_std_i = lo + 0.5 * (hi - lo) * (log_std_i + 1)
            std_i = log_std_i.exp()
            
            dist_i = Normal(mu_i, std_i)
            if given_actions is None:
                # a_i = dist_i.rsample() if rsample else dist_i.sample()
                # print(f"dist_i: {dist_i.loc}")
                a_i = dist_i.loc
            else:
                a_i = given_actions[:, i].unsqueeze(-1)      # use provided action
            lp_i = dist_i.log_prob(a_i).squeeze(-1)         # [B]

            actions.append(a_i)
            log_probs.append(lp_i)
            mus.append(mu_i)
            stds.append(std_i)

        # stitch back
        actions  = torch.cat(actions, dim=-1)               # [B x action_dim]
        mus       = torch.cat(mus,      dim=-1)             # [B x action_dim]
        stds      = torch.cat(stds,     dim=-1)             # [B x action_dim]
        log_prob  = torch.stack(log_probs, dim=1).sum(dim=1, keepdims=True)  # [B, 1]

        # for diagnostics
        self.outputs['mu']  = mus
        self.outputs['std'] = stds

        return actions, log_prob
    
class AutoregressiveDiscreteActor(nn.Module):
    def __init__(self, obs_dim, n_action_dim, hidden_dim, hidden_depth, n_action=5):
        """
        obs_dim:       dimension of the observation vector
        n_action_dim:  number of discrete action dimensions
        hidden_dim:    width of hidden layers
        hidden_depth:  number of hidden layers in each MLP
        n_action:      cardinality of each discrete dimension (here 5)
        """
        super().__init__()
        self.obs_dim       = obs_dim
        self.n_action_dim  = n_action_dim
        self.n_action      = n_action

        # one small net per action-dimension, outputting `n_action` logits
        # input size for net[i] = obs_dim + i * n_action
        self.logit_nets = nn.ModuleList([
            util.mlp(obs_dim + i * n_action,
                     hidden_dim,
                     n_action,
                     hidden_depth)
            for i in range(n_action_dim)
        ])

        self.outputs = {}
        self.apply(util.weight_init)

    def _sample_and_logprob(self, obs, given_actions=None):
        """
        If given_actions is None: sample autoregressively.
        Otherwise, compute log-prob of provided actions.
        Returns:
          actions:  [B x n_action_dim]  ints in [0..n_action-1]
          log_prob: [B]                 log p(a|s)
        """
        B = obs.size(0)
        action_list  = []
        logp_list    = []
        prev_onehots = []

        for i, net in enumerate(self.logit_nets):
            # build input: first just obs, then obs + one‐hots of previous actions
            if i == 0:
                inp = obs                                          # [B x obs_dim]
            else:
                prev = torch.cat(prev_onehots, dim=-1)             # [B x (i * n_action)]
                inp  = torch.cat([obs, prev], dim=-1)              # [B x (obs_dim + i*n_action)]

            logits = net(inp)                                      # [B x n_action]
            dist   = Categorical(logits=logits)

            if given_actions is None:
                a_i = dist.sample()                                # [B]
            else:
                a_i = given_actions[:, i]                          # [B]

            lp_i = dist.log_prob(a_i)                              # [B]
            # print(f'a_i: {a_i.shape, a_i, a_i.dtype}')
            # print(f'self.n_action: {self.n_action, type(self.n_action)}')  
            a_i_oh = F.one_hot(a_i.to(torch.int64), num_classes = self.n_action).float()         # [B x n_action]

            action_list.append(a_i)
            logp_list.append(lp_i)
            prev_onehots.append(a_i_oh)

        actions  = torch.stack(action_list, dim=1)                # [B x n_action_dim]
        log_prob = torch.stack(logp_list, dim=1).sum(dim=1, keepdims=True)       # [B]

        # optional: store the final logits for inspection
        # self.outputs['logits'] = torch.stack(
        #     [net(inp) for net, inp in zip(self.logit_nets,
        #                                   [obs if i==0 else torch.cat([obs]+prev_onehots[:i],dim=-1)
        #                                    for i in range(self.n_action_dim)])],
        #     dim=1
        # )  # [B x n_action_dim x n_action]

        return actions, log_prob

    def forward(self, obs):
        """Sample actions and return log-prob (for e.g. policy gradients)."""
        return self._sample_and_logprob(obs, given_actions=None)

    def select_action(self, obs):
        """Just sample one action (no grad)."""
        with torch.no_grad():
            actions, _ = self._sample_and_logprob(obs, given_actions=None)
            return actions

    def log_prob(self, obs, actions):
        """
        Compute log-prob of given discrete actions:
          obs:     [B x obs_dim]
          actions: [B x n_action_dim]  ints
        Returns: [B]
        """
        _, logp = self._sample_and_logprob(obs, given_actions=actions)
        return logp