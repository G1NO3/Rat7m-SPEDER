import copy
import math
import logging
import dataclasses
from collections import OrderedDict
import typing as tp
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from hydra.core.config_store import ConfigStore
import omegaconf
from utils.util import unpack_batch
from utils.buffer import ReplayBuffer
from agent.hilp import utilss
from utils.util import MLP, DoubleMLP
import typing as tp
# from url_benchmark import utils
# from url_benchmark.in_memory_replay_buffer import ReplayBuffer
# from .ddpg import MetaDict, make_aug_encoder
# from .fb_modules import Actor, DiagGaussianActor, ForwardMap, BackwardMap, mlp, OnlineCov
from agent.sac.actor import DiagGaussianActor

# from url_benchmark.dmc import TimeStep

MetaDict = tp.Mapping[str, np.ndarray]
logger = logging.getLogger(__name__)

    
@dataclasses.dataclass
class SFAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.sf.SFAgent"
    name: str = "sf"
    obs_type: str = 'keypoint'  # to be specified later
    image_wh: int = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = 'cuda:0'  # ${device}
    lr: float = 1e-4
    lr_coef: float = 1
    sf_target_tau: float = 0.01  # 0.001-0.01
    update_every_steps: int = 1
    use_tb: bool = True  # ${use_tb}
    use_wandb: bool = False  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING  # ???  # to be specified later
    num_inference_steps: int = 10000
    hidden_dim: int = 1024   # 128, 2048
    phi_hidden_dim: int = 128   # 128, 2048
    feature_dim: int = 128   # 128, 1024
    z_dim: int = 64  # 30-200
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)"  # 0,  0.1, 0.2
    stddev_clip: float = 0.3  # 1
    update_z_every_step: int = 300
    nstep: int = 1
    batch_size: int = 16
    init_sf: bool = True
    update_encoder: bool = omegaconf.II("update_encoder")  # ${update_encoder}
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)  # param for DiagGaussianActor
    temp: float = 1  # temperature for DiagGaussianActor
    boltzmann: bool = True  # set to true for DiagGaussianActor
    debug: bool = False
    preprocess: bool = True
    num_sf_updates: int = 1
    feature_learner: str = "hilp"
    mix_ratio: float = 0.5
    q_loss: bool = True
    update_cov_every_step: int = 1000
    add_trunk: bool = False
    sf_reg: float = 1

    feature_type: str = 'state'  # 'state', 'diff', 'concat'
    hilp_discount: float = 0.9
    hilp_expectile: float = 0.5
    def __init__(self, **kwargs):
        # 遍历 dataclass 字段
        for f in dataclasses.fields(self):
            if f.name in kwargs:
                setattr(self, f.name, kwargs.pop(f.name))
            else:
                setattr(self, f.name, f.default)
        # 其余的 kwargs 就存在 extra_args 里
        self.extra_args = kwargs


# cs = ConfigStore.instance()
# cs.store(group="agent", name="sf", node=SFAgentConfig)
class _L2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = math.sqrt(self.dim) * F.normalize(x, dim=1)
        return y
def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")
def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    """Provides a sequence of linear layers and non-linearities
    providing a sequence of dimension for the neurons, or name of
    the non-linearities
    Eg: mlp(10, 12, "relu", 15) returns:
    Sequential(Linear(10, 12), ReLU(), Linear(12, 15))
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert isinstance(layer, int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)

class ForwardMap(nn.Module):
    """ forward representation class"""

    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess
        if self.preprocess:
            self.obs_action_net = mlp(self.obs_dim + self.action_dim, feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim + self.action_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim
        seq = [feature_dim, hidden_dim, "irelu", self.z_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)
        self.apply(utilss.weight_init)



        self.log_scale = nn.Parameter(torch.tensor(-2.3025851))
        with torch.no_grad():
            last1 = [m for m in self.F1.modules() if isinstance(m, nn.Linear)][-1]
            last2 = [m for m in self.F2.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last1.weight); nn.init.zeros_(last1.bias)
            nn.init.zeros_(last2.weight); nn.init.zeros_(last2.bias)

    def forward(self, obs, z, action):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs_action, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z, action], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        F1 = self.F1(h)
        F2 = self.F2(h)


        
        scale = torch.exp(self.log_scale)
        F1 = torch.tanh(F1) * scale * self.z_dim
        F2 = torch.tanh(F2) * scale * self.z_dim
        return F1, F2


class FeatureLearner(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        self.feature_net: nn.Module = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utilss.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        return None


class Identity(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.feature_net = nn.Identity()


class HILP(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, cfg) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.z_dim = z_dim
        self.cfg = cfg

        if self.cfg.feature_type != 'concat':
            feature_dim = z_dim
        else:
            assert z_dim % 2 == 0
            feature_dim = z_dim // 2

        layers = [obs_dim, hidden_dim, "relu", feature_dim]
        print('layers', layers)
        self.phi1 = mlp(*layers)
        self.phi2 = mlp(*layers)
        self.target_phi1 = mlp(*layers)
        self.target_phi2 = mlp(*layers)
        self.target_phi1.load_state_dict(self.phi1.state_dict())
        self.target_phi2.load_state_dict(self.phi2.state_dict())

        self.apply(utilss.weight_init)

        # Define a running mean and std
        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_std', torch.ones(feature_dim))

    def feature_net(self, obs):
        phi = self.phi1(obs)
        phi = phi - self.running_mean
        # phi = (phi - self.running_mean) / (self.running_std + 1e-6)
        # phi = F.normalize(phi, dim=-1)
        return phi

    def value(self, obs: torch.Tensor, goals: torch.Tensor, is_target: bool = False):
        if is_target:
            phi1 = self.target_phi1
            phi2 = self.target_phi2
        else:
            phi1 = self.phi1
            phi2 = self.phi2

        phi1_s = phi1(obs)
        phi1_g = phi1(goals)

        phi2_s = phi2(obs)
        phi2_g = phi2(goals)

        squared_dist1 = ((phi1_s - phi1_g) ** 2).sum(dim=-1)
        v1 = -torch.sqrt(torch.clamp(squared_dist1, min=1e-6))
        squared_dist2 = ((phi2_s - phi2_g) ** 2).sum(dim=-1)
        v2 = -torch.sqrt(torch.clamp(squared_dist2, min=1e-6))

        if is_target:
            v1 = v1.detach()
            v2 = v2.detach()

        return v1, v2

    def expectile_loss(self, adv, diff, expectile=0.7):
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        goals = future_obs
        rewards = (torch.linalg.norm(obs - goals, dim=-1) < 1e-6).float()
        # rewards = - torch.linalg.norm(obs - goals, dim=-1) 
        masks = 1.0 - rewards
        # masks = 1.0 - (torch.linalg.norm(next_obs - goals, dim=-1) < 1e-6).float()
        rewards = rewards - 1.0

        next_v1, next_v2 = self.value(next_obs, goals, is_target=True)
        next_v = torch.minimum(next_v1, next_v2)
        q = rewards + self.cfg.hilp_discount * masks * next_v

        v1_t, v2_t = self.value(obs, goals, is_target=True)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = rewards + self.cfg.hilp_discount * masks * next_v1
        q2 = rewards + self.cfg.hilp_discount * masks * next_v2
        v1, v2 = self.value(obs, goals, is_target=False)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.cfg.hilp_expectile).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.cfg.hilp_expectile).mean()
        value_loss = value_loss1 + value_loss2

        utilss.soft_update_params(self.phi1, self.target_phi1, 0.005)
        utilss.soft_update_params(self.phi2, self.target_phi2, 0.005)

        with torch.no_grad():
            phi1 = self.phi1(obs)
            self.running_mean = 0.995 * self.running_mean + 0.005 * phi1.mean(dim=0)
            self.running_std = 0.995 * self.running_std + 0.005 * phi1.std(dim=0)

        return value_loss, {
            'hilp/value_loss': value_loss,
            'hilp/v_mean': v.mean(),
            'hilp/v_max': v.max(),
            'hilp/v_min': v.min(),
            'hilp/abs_adv_mean': torch.abs(adv).mean(),
            'hilp/adv_mean': adv.mean(),
            'hilp/adv_max': adv.max(),
            'hilp/adv_min': adv.min(),
            'hilp/accept_prob': (adv >= 0).float().mean(),
        }


class Laplacian(FeatureLearner):
    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del action
        del future_obs
        phi = self.feature_net(obs)
        next_phi = self.feature_net(next_obs)
        loss = (phi - next_phi).pow(2).mean()
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        return loss


class ContrastiveFeature(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utilss.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del action
        del next_obs
        assert future_obs is not None
        phi = self.feature_net(obs)
        future_mu = self.mu_net(future_obs)
        phi = F.normalize(phi, dim=1)
        future_mu = F.normalize(future_mu, dim=1)
        logits = torch.einsum('sd, td-> st', phi, future_mu)  # batch x batch
        I = torch.eye(*logits.size(), device=logits.device)
        off_diag = ~I.bool()
        logits_off_diag = logits[off_diag].reshape(logits.shape[0], logits.shape[0] - 1)
        loss = - logits.diag() + torch.logsumexp(logits_off_diag, dim=1)
        loss = loss.mean()
        return loss


class ContrastiveFeaturev2(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utilss.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del action
        del next_obs
        assert future_obs is not None
        future_phi = self.feature_net(future_obs)
        mu = self.mu_net(obs)
        future_phi = F.normalize(future_phi, dim=1)
        mu = F.normalize(mu, dim=1)
        logits = torch.einsum('sd, td-> st', mu, future_phi)  # batch x batch
        I = torch.eye(*logits.size(), device=logits.device)
        off_diag = ~I.bool()
        logits_off_diag = logits[off_diag].reshape(logits.shape[0], logits.shape[0] - 1)
        loss = - logits.diag() + torch.logsumexp(logits_off_diag, dim=1)
        loss = loss.mean()
        return loss


class ICM(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.inverse_dynamic_net = mlp(2 * z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', action_dim, 'tanh')
        self.apply(utilss.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        next_phi = self.feature_net(next_obs)
        predicted_action = self.inverse_dynamic_net(torch.cat([phi, next_phi], dim=-1))
        backward_error = (action - predicted_action).pow(2).mean()
        icm_loss = backward_error
        return icm_loss


class TransitionModel(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', obs_dim)
        self.apply(utilss.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (predicted_next_obs - next_obs).pow(2).mean()
        return forward_error


class TransitionLatentModel(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', z_dim)
        self.target_feature_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utilss.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        with torch.no_grad():
            next_phi = self.target_feature_net(next_obs)
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (predicted_next_obs - next_phi.detach()).pow(2).mean()
        utilss.soft_update_params(self.feature_net, self.target_feature_net, 0.01)

        return forward_error


class AutoEncoder(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.decoder = mlp(z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', obs_dim)
        self.apply(utilss.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        del next_obs
        del action
        phi = self.feature_net(obs)
        predicted_obs = self.decoder(phi)
        reconstruction_error = (predicted_obs - obs).pow(2).mean()
        return reconstruction_error


class SVDSR(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.target_feature_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.target_mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(utilss.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        mu = self.mu_net(next_obs)
        SR = torch.einsum("sd, td -> st", phi, mu)
        with torch.no_grad():
            target_phi = self.target_feature_net(next_obs)
            target_mu = self.target_mu_net(next_obs)
            target_SR = torch.einsum("sd, td -> st", target_phi, target_mu)

        I = torch.eye(*SR.size(), device=SR.device)
        off_diag = ~I.bool()
        loss = - 2 * SR.diag().mean() + (SR - 0.99 * target_SR.detach())[off_diag].pow(2).mean()

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        utilss.soft_update_params(self.feature_net, self.target_feature_net, 0.01)
        utilss.soft_update_params(self.mu_net, self.target_mu_net, 0.01)

        return loss


class SVDSRv2(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.target_feature_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.target_mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(utilss.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(next_obs)
        mu = self.mu_net(obs)
        SR = torch.einsum("sd, td -> st", mu, phi)
        with torch.no_grad():
            target_phi = self.target_feature_net(next_obs)
            target_mu = self.target_mu_net(next_obs)
            target_SR = torch.einsum("sd, td -> st", target_mu, target_phi)

        I = torch.eye(*SR.size(), device=SR.device)
        off_diag = ~I.bool()
        loss = - 2 * SR.diag().mean() + (SR - 0.98 * target_SR.detach())[off_diag].pow(2).mean()

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        utilss.soft_update_params(self.feature_net, self.target_feature_net, 0.01)
        utilss.soft_update_params(self.mu_net, self.target_mu_net, 0.01)

        return loss


class SVDP(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim + action_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(utilss.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(next_obs)
        mu = self.mu_net(torch.cat([obs, action], dim=1))
        P = torch.einsum("sd, td -> st", mu, phi)
        I = torch.eye(*P.size(), device=P.device)
        off_diag = ~I.bool()
        loss = - 2 * P.diag().mean() + P[off_diag].pow(2).mean()

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        return loss


class HILPAgent:

    def __init__(self,            
            state_dim,
            action_dim,
            lr=1e-3,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=256,
            **kwargs):
        cfg = SFAgentConfig(**kwargs)
        self.cfg = cfg
        self.cfg.lr = lr
        self.cfg.hidden_dim = hidden_dim
        self.cfg.hilp_discount = discount
        self.cfg.action_shape = (action_dim,)
        self.cfg.obs_shape = (state_dim,)
        self.cfg.obs_dim = self.obs_dim = self.state_dim = state_dim
        self.cfg.action_dim = self.action_dim = action_dim
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]
        self.solved_meta: tp.Any = None
        self.steps = 0

        # models
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)).to(self.cfg.device)
        # create the network

        self.actor: nn.Module = DiagGaussianActor(obs_dim=self.obs_dim+self.cfg.hidden_dim, action_dim=self.action_dim,
                                                  hidden_dim=cfg.hidden_dim, hidden_depth=2,
                                                    log_std_bounds=cfg.log_std_bounds).to(cfg.device)
        self.successor_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                        cfg.feature_dim, cfg.hidden_dim,
                                        preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        # build up the target network
        self.successor_target_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                               cfg.feature_dim, cfg.hidden_dim,
                                               preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)

        learner = dict(icm=ICM, transition=TransitionModel, latent=TransitionLatentModel,
                       contrastive=ContrastiveFeature, autoencoder=AutoEncoder, lap=Laplacian,
                       random=FeatureLearner, svd_sr=SVDSR, svd_p=SVDP,
                       contrastivev2=ContrastiveFeaturev2, svd_srv2=SVDSRv2,
                       identity=Identity, hilp=HILP)[self.cfg.feature_learner]
        extra_kwargs = dict()
        if self.cfg.feature_learner == 'hilp':
            extra_kwargs = dict(
                cfg=self.cfg,
            )
        self.feature_learner = learner(self.obs_dim, self.action_dim, cfg.z_dim, cfg.phi_hidden_dim, **extra_kwargs).to(cfg.device)

        # load the weights into the target networks
        self.successor_target_net.load_state_dict(self.successor_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.sf_opt = torch.optim.Adam(self.successor_net.parameters(), lr=cfg.lr)
        self.phi_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.feature_learner not in ["random", "identity"]:
            self.phi_opt = torch.optim.Adam(self.feature_learner.parameters(), lr=cfg.lr_coef * cfg.lr)
        self.successor_target_net.train()

        self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)

    def init_from(self, other) -> None:
        # copy parameters over
        names = ["encoder", "actor"]
        if self.cfg.init_sf:
            names += ["successor_net", "feature_learner", "successor_target_net"]
        for name in names:
            utilss.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def get_goal_meta(self, goal_array: np.ndarray, obs_array: np.ndarray = None) -> MetaDict:
        assert self.cfg.feature_learner == 'hilp'

        obs = torch.tensor(obs_array).unsqueeze(0).to(self.cfg.device)
        desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.cfg.device)

        with torch.no_grad():
            obs = self.encoder(obs)
            desired_goal = self.encoder(desired_goal)

        with torch.no_grad():
            z_g = self.feature_learner.feature_net(desired_goal)
            z_s = self.feature_learner.feature_net(obs)

        z = (z_g - z_s)
        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor):
        with torch.no_grad():
            obs = self.encoder(obs)
            next_obs = self.encoder(next_obs)

        with torch.no_grad():
            if self.cfg.feature_type == 'state':
                phi = self.feature_learner.feature_net(obs)
            elif self.cfg.feature_type == 'diff':
                phi = self.feature_learner.feature_net(next_obs) - self.feature_learner.feature_net(obs)
            else:
                phi = torch.cat([self.feature_learner.feature_net(obs), self.feature_learner.feature_net(next_obs)], dim=-1)
        z = torch.linalg.lstsq(phi, reward).solution

        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
        meta = OrderedDict()
        meta['z'] = z.squeeze().cpu().numpy()
        return meta

    def sample_z(self, size):
        gaussian_rdv = torch.randn((size, self.cfg.z_dim), dtype=torch.float32)
        z = math.sqrt(self.cfg.z_dim) * F.normalize(gaussian_rdv, dim=1)
        return z

    def init_meta(self) -> MetaDict:
        if self.solved_meta is not None:
            print('solved_meta')
            return self.solved_meta
        else:
            z = self.sample_z(1)
            z = z.squeeze().numpy()
            meta = OrderedDict()
            meta['z'] = z
        return meta

    # pylint: disable=unused-argument
    def update_meta(
            self,
            meta: MetaDict,
            global_step: int,
            time_step,
            finetune: bool = False,
            replay_loader: tp.Optional[ReplayBuffer] = None
    ) -> MetaDict:
        if global_step % self.cfg.update_z_every_step == 0:
            return self.init_meta()
        return meta

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta['z'], device=self.cfg.device).unsqueeze(0)  # type: ignore
        if self.cfg.boltzmann:
            dist = self.actor(h, z)
        else:
            stddev = utilss.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(h, z, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_sf(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: tp.Optional[torch.Tensor],
        z: torch.Tensor,
        step: int
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():

            dist = self.actor(torch.concat([next_obs, z], dim=-1))
            next_action = dist.sample()

            next_F1, next_F2 = self.successor_target_net(next_obs, z, next_action)  # batch x z_dim
            if self.cfg.feature_type == 'state':
                target_phi = self.feature_learner.feature_net(next_obs).detach()  # batch x z_dim
            elif self.cfg.feature_type == 'diff':
                target_phi = self.feature_learner.feature_net(next_obs).detach() - self.feature_learner.feature_net(obs).detach()
            else:
                target_phi = torch.cat([self.feature_learner.feature_net(obs).detach(), self.feature_learner.feature_net(next_obs).detach()], dim=-1)
            next_Q1, next_Q2 = [torch.einsum('sd, sd -> s', next_Fi, z) for next_Fi in [next_F1, next_F2]]
            next_F = torch.where((next_Q1 < next_Q2).reshape(-1, 1), next_F1, next_F2)
            target_F = target_phi + discount * next_F

        F1, F2 = self.successor_net(obs, z, action)
        if self.cfg.q_loss:
            Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
            target_Q = torch.einsum('sd, sd -> s', target_F, z)
            sf_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        else:
            sf_loss = F.mse_loss(F1, target_F) + F.mse_loss(F2, target_F)
        # reg_loss = (F1.pow(2).sum(dim=-1).mean() + F2.pow(2).sum(dim=-1).mean())
        # sf_loss = sf_loss + self.cfg.sf_reg * reg_loss
        sf_loss = sf_loss

        # compute feature loss
        if self.cfg.feature_learner == 'hilp':
            phi_loss, info = self.feature_learner(obs=obs, action=action, next_obs=next_obs, future_obs=future_obs)
        else:
            phi_loss = self.feature_learner(obs=obs, action=action, next_obs=next_obs, future_obs=future_obs)
            info = None

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['target_F'] = target_F.mean().item()
            metrics['F1'] = F1.mean().item()
            metrics['phi'] = target_phi.mean().item()
            metrics['phi_norm'] = torch.norm(target_phi, dim=-1).mean().item()
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['sf_loss'] = sf_loss.item()
            metrics['F_log_scale'] = self.successor_net.log_scale.item()
            if phi_loss is not None:
                metrics['phi_loss'] = phi_loss.item()

            if isinstance(self.sf_opt, torch.optim.Adam):
                metrics["sf_opt_lr"] = self.sf_opt.param_groups[0]["lr"]

            if info is not None:
                for key, val in info.items():
                    metrics[key] = val.item()

        # optimize SF
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.sf_opt.zero_grad(set_to_none=True)
        if self.phi_opt is not None:
            self.phi_opt.zero_grad(set_to_none=True)
            phi_loss.backward(retain_graph=True)
        sf_loss.backward()
        self.sf_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        if self.phi_opt is not None:
            self.phi_opt.step()

        return metrics

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        dist = self.actor(torch.concat([obs, z], dim=-1))
        action = dist.rsample()

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.successor_net(obs, z, action)
        Q1 = torch.einsum('sd, sd -> s', F1, z)
        Q2 = torch.einsum('sd, sd -> s', F2, z)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.cfg.temp * log_prob - Q).mean() if self.cfg.boltzmann else -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()

        return metrics

    def aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.aug(obs)
        return self.encoder(obs)


    def train(self, replay_loader: ReplayBuffer, batch_size, seq_len):
        metrics: tp.Dict[str, float] = {}
        self.steps += 1
        batch = replay_loader.sample_sequence(batch_size, seq_len)
        expert_state, expert_action, expert_next_state, expert_reward, expert_done, expert_task, expert_next_task = unpack_batch(batch)
        B, T = expert_state.shape[0], expert_state.shape[1]
        assert expert_state.shape == (B, T, self.state_dim)
        assert expert_action.shape == (B, T, self.action_dim)
        assert expert_next_state.shape == (B, T, self.state_dim)
        future_expert_state = expert_state[:, -1:, :].repeat(1, T, 1)

        z = self.sample_z(B*T).to(self.cfg.device)
        if not z.shape[-1] == self.cfg.z_dim:
            raise RuntimeError("There's something wrong with the logic here")

        processed_state = expert_state.reshape(B * T, self.state_dim)
        processed_next_state = expert_next_state.reshape(B * T, self.state_dim)
        processed_future_state = future_expert_state.reshape(B * T, self.state_dim)
        processed_action = expert_action.reshape(B * T, self.action_dim)
        if self.cfg.mix_ratio > 0:
            perm = torch.randperm(B*T)
            with torch.no_grad():
                if self.cfg.feature_type == 'state':
                    desired_obs = processed_next_state[perm]
                    phi = self.feature_learner.feature_net(desired_obs)
                elif self.cfg.feature_type == 'diff':
                    desired_obs = processed_state[perm]
                    desired_next_obs = processed_next_state[perm]
                    phi = self.feature_learner.feature_net(desired_next_obs) - self.feature_learner.feature_net(desired_obs)
                else:
                    desired_obs = processed_state[perm]
                    desired_next_obs = processed_next_state[perm]
                    phi = torch.cat([self.feature_learner.feature_net(desired_obs), self.feature_learner.feature_net(desired_next_obs)], dim=-1)
            # compute inverse of cov of phi
            cov = torch.matmul(phi.T, phi) / phi.shape[0]
            inv_cov = torch.linalg.pinv(cov)

            mix_idxs: tp.Any = np.where(np.random.uniform(size=B*T) < self.cfg.mix_ratio)[0]
            with torch.no_grad():
                new_z = phi[mix_idxs]

            new_z = torch.matmul(new_z, inv_cov)  # batch_size x z_dim
            new_z = math.sqrt(self.cfg.z_dim) * F.normalize(new_z, dim=1)
            z[mix_idxs] = new_z

        critic_info = self.update_sf(obs=processed_state, action=processed_action, discount=self.cfg.hilp_discount, 
                                     next_obs=processed_next_state, future_obs=processed_future_state, z=z, step=self.steps)

        # update actor
        actor_info = self.update_actor(processed_state.detach(), z, self.steps)

        metrics.update(critic_info)
        metrics.update(actor_info)
        # update critic target
        utilss.soft_update_params(self.successor_net, self.successor_target_net, self.cfg.sf_target_tau)

        return metrics
    def state_dict(self):
        state = {
            'state_processor': self.state_processor.state_dict(),
            'actor': self.actor.state_dict(),
            'successor_net': self.successor_net.state_dict(),
            'successor_target_net': self.successor_target_net.state_dict(),
            'feature_learner': self.feature_learner.state_dict(),
        }
        return state
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.successor_net.load_state_dict(state_dict['successor_net'])
        self.successor_target_net.load_state_dict(state_dict['successor_target_net'])
        self.feature_learner.load_state_dict(state_dict['feature_learner'])
    def n_params(self):
        modules = [self.state_processor, self.actor, self.successor_net, self.feature_learner]
        for m in modules:
            for n, p in m.named_parameters():
                if p.requires_grad:
                    print(f"{n}: {p.numel()}")
        n = sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)
        return n
    def action_loglikelihood(self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor):
        F1, F2 = self.successor_net(obs, z, action)
        Q1 = torch.einsum('sd, sd -> s', F1, z)
        Q2 = torch.einsum('sd, sd -> s', F2, z)
        Q = torch.min(Q1, Q2)
        return Q


class NewHILPAgent:

    def __init__(self,            
            state_dim,
            action_dim,
            lr=1e-3,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=256,
            **kwargs):
        cfg = SFAgentConfig(**kwargs)
        self.cfg = cfg
        self.cfg.lr = lr
        self.cfg.hidden_dim = hidden_dim
        self.cfg.hilp_discount = discount
        self.cfg.action_shape = (action_dim,)
        self.cfg.obs_shape = (state_dim,)
        self.cfg.obs_dim = self.obs_dim = self.state_dim = state_dim
        self.cfg.action_dim = self.action_dim = action_dim
        self.hidden_dim = self.cfg.hidden_dim = hidden_dim
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]
        self.solved_meta: tp.Any = None
        self.steps = 0

        # models
        # create the network

        self.actor: nn.Module = DiagGaussianActor(obs_dim=self.obs_dim+self.cfg.hidden_dim, action_dim=self.action_dim,
                                                  hidden_dim=cfg.hidden_dim, hidden_depth=2,
                                                    log_std_bounds=cfg.log_std_bounds).to(cfg.device)
        
        # self.successor_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
        #                                 cfg.feature_dim, cfg.hidden_dim,
        #                                 preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        self.successor_net = DoubleMLP(input_dim=self.obs_dim + self.action_dim + self.hidden_dim, hidden_dim=cfg.hidden_dim,
                                       output_dim=cfg.z_dim, hidden_depth=1).to(cfg.device)
        # build up the target network
        # self.successor_target_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
        #                                        cfg.feature_dim, cfg.hidden_dim,
        #                                        preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        self.successor_target_net = DoubleMLP(input_dim=self.obs_dim + self.action_dim + self.hidden_dim, hidden_dim=cfg.hidden_dim,
                                        output_dim=cfg.z_dim, hidden_depth=1).to(cfg.device)

        learner = dict(icm=ICM, transition=TransitionModel, latent=TransitionLatentModel,
                       contrastive=ContrastiveFeature, autoencoder=AutoEncoder, lap=Laplacian,
                       random=FeatureLearner, svd_sr=SVDSR, svd_p=SVDP,
                       contrastivev2=ContrastiveFeaturev2, svd_srv2=SVDSRv2,
                       identity=Identity, hilp=NewHILP)[self.cfg.feature_learner]
        extra_kwargs = dict()
        if self.cfg.feature_learner == 'hilp':
            extra_kwargs = dict(
                cfg=self.cfg,
            )
        self.feature_learner = learner(self.obs_dim, self.action_dim, cfg.z_dim, cfg.phi_hidden_dim, **extra_kwargs).to(cfg.device)

        # load the weights into the target networks
        self.successor_target_net.load_state_dict(self.successor_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.sf_opt = torch.optim.Adam(self.successor_net.parameters(), lr=cfg.lr)
        self.phi_opt = torch.optim.Adam(self.feature_learner.parameters(), lr=cfg.lr_coef * cfg.lr)
        self.successor_target_net.train()

        self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)

        # self.training = training
        # for net in [self.encoder, self.actor, self.successor_net]:
        #     net.train(training)
        # if self.phi_opt is not None:
        #     self.feature_learner.train()

    def init_from(self, other) -> None:
        # copy parameters over
        names = ["encoder", "actor"]
        if self.cfg.init_sf:
            names += ["successor_net", "feature_learner", "successor_target_net"]
        for name in names:
            utilss.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def get_goal_meta(self, goal_array: np.ndarray, obs_array: np.ndarray = None) -> MetaDict:
        assert self.cfg.feature_learner == 'hilp'

        obs = torch.tensor(obs_array).unsqueeze(0).to(self.cfg.device)
        desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.cfg.device)

        with torch.no_grad():
            obs = self.encoder(obs)
            desired_goal = self.encoder(desired_goal)

        with torch.no_grad():
            z_g = self.feature_learner.feature_net(desired_goal)
            z_s = self.feature_learner.feature_net(obs)

        z = (z_g - z_s)
        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor):
        with torch.no_grad():
            obs = self.encoder(obs)
            next_obs = self.encoder(next_obs)

        with torch.no_grad():
            if self.cfg.feature_type == 'state':
                phi = self.feature_learner.feature_net(obs)
            elif self.cfg.feature_type == 'diff':
                phi = self.feature_learner.feature_net(next_obs) - self.feature_learner.feature_net(obs)
            else:
                phi = torch.cat([self.feature_learner.feature_net(obs), self.feature_learner.feature_net(next_obs)], dim=-1)
        z = torch.linalg.lstsq(phi, reward).solution

        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
        meta = OrderedDict()
        meta['z'] = z.squeeze().cpu().numpy()
        return meta

    def sample_z(self, size):
        gaussian_rdv = torch.randn((size, self.cfg.z_dim), dtype=torch.float32)
        z = math.sqrt(self.cfg.z_dim) * F.normalize(gaussian_rdv, dim=1)
        return z

    def init_meta(self) -> MetaDict:
        if self.solved_meta is not None:
            print('solved_meta')
            return self.solved_meta
        else:
            z = self.sample_z(1)
            z = z.squeeze().numpy()
            meta = OrderedDict()
            meta['z'] = z
        return meta

    # pylint: disable=unused-argument
    def update_meta(
            self,
            meta: MetaDict,
            global_step: int,
            time_step,
            finetune: bool = False,
            replay_loader: tp.Optional[ReplayBuffer] = None
    ) -> MetaDict:
        if global_step % self.cfg.update_z_every_step == 0:
            return self.init_meta()
        return meta

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta['z'], device=self.cfg.device).unsqueeze(0)  # type: ignore
        if self.cfg.boltzmann:
            dist = self.actor(h, z)
        else:
            stddev = utilss.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(h, z, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_sf(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: tp.Optional[torch.Tensor],
        z: torch.Tensor,
        step: int
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():

            dist = self.actor(torch.concat([next_obs, z], dim=-1))
            next_action = dist.sample()

            next_F1, next_F2 = self.successor_target_net(torch.concat([next_obs, z, next_action], dim=-1))  # batch x z_dim
            if self.cfg.feature_type == 'state':
                target_phi = self.feature_learner.feature_net(next_obs).detach()  # batch x z_dim
            elif self.cfg.feature_type == 'diff':
                target_phi = self.feature_learner.feature_net(next_obs).detach() - self.feature_learner.feature_net(obs).detach()
            else:
                target_phi = torch.cat([self.feature_learner.feature_net(obs).detach(), self.feature_learner.feature_net(next_obs).detach()], dim=-1)
            next_Q1, next_Q2 = [torch.einsum('sd, sd -> s', next_Fi, z) for next_Fi in [next_F1, next_F2]]
            next_F = torch.where((next_Q1 < next_Q2).reshape(-1, 1), next_F1, next_F2)
            target_F = target_phi + discount * next_F

        # DoubleMLP adaptation
        # F1, F2 = self.successor_net(obs, z, action)
        F1, F2 = self.successor_net(torch.cat([obs, action, z], dim=-1))
        if self.cfg.q_loss:
            Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
            target_Q = torch.einsum('sd, sd -> s', target_F, z)
            sf_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        else:
            sf_loss = F.mse_loss(F1, target_F) + F.mse_loss(F2, target_F)
        reg_loss = (F1.pow(2).sum(dim=-1).mean() + F2.pow(2).sum(dim=-1).mean())
        sf_loss = sf_loss + self.cfg.sf_reg * reg_loss

        # compute feature loss
        if self.cfg.feature_learner == 'hilp':
            phi_loss, info = self.feature_learner(obs=obs, action=action, next_obs=next_obs, future_obs=future_obs)
        else:
            phi_loss = self.feature_learner(obs=obs, action=action, next_obs=next_obs, future_obs=future_obs)
            info = None

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['target_F'] = target_F.mean().item()
            metrics['F1'] = F1.mean().item()
            metrics['phi'] = target_phi.mean().item()
            metrics['phi_norm'] = torch.norm(target_phi, dim=-1).mean().item()
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['sf_loss'] = sf_loss.item()
            if phi_loss is not None:
                metrics['phi_loss'] = phi_loss.item()

            if isinstance(self.sf_opt, torch.optim.Adam):
                metrics["sf_opt_lr"] = self.sf_opt.param_groups[0]["lr"]

            if info is not None:
                for key, val in info.items():
                    metrics[key] = val.item()

        # optimize SF
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.sf_opt.zero_grad(set_to_none=True)
        if self.phi_opt is not None:
            self.phi_opt.zero_grad(set_to_none=True)
            phi_loss.backward(retain_graph=True)
        sf_loss.backward()
        self.sf_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        if self.phi_opt is not None:
            self.phi_opt.step()

        return metrics

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        if self.cfg.boltzmann:
            dist = self.actor(torch.concat([obs, z], dim=-1))
            action = dist.rsample()
        else:
            stddev = utilss.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(obs, z, stddev)
            action = dist.sample(clip=self.cfg.stddev_clip)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # F1, F2 = self.successor_net(obs, z, action)
        F1, F2 = self.successor_net(torch.cat([obs, action, z], dim=-1))
        Q1 = torch.einsum('sd, sd -> s', F1, z)
        Q2 = torch.einsum('sd, sd -> s', F2, z)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.cfg.temp * log_prob - Q).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()

        return metrics

    def aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.aug(obs)
        return self.encoder(obs)


    def train(self, replay_loader: ReplayBuffer, batch_size, seq_len):
        metrics: tp.Dict[str, float] = {}
        self.steps += 1
        batch = replay_loader.sample_sequence(batch_size, seq_len)
        expert_state, expert_action, expert_next_state, expert_reward, expert_done, expert_task, expert_next_task = unpack_batch(batch)
        B, T = expert_state.shape[0], expert_state.shape[1]
        assert expert_state.shape == (B, T, self.state_dim)
        assert expert_action.shape == (B, T, self.action_dim)
        assert expert_next_state.shape == (B, T, self.state_dim)
        future_expert_state = expert_state[:, -1:, :].repeat(1, T, 1)

        z = self.sample_z(B*T).to(self.cfg.device)
        if not z.shape[-1] == self.cfg.z_dim:
            raise RuntimeError("There's something wrong with the logic here")

        processed_state = expert_state.reshape(B * T, self.state_dim)
        processed_next_state = expert_next_state.reshape(B * T, self.state_dim)
        processed_future_state = future_expert_state.reshape(B * T, self.state_dim)
        processed_action = expert_action.reshape(B * T, self.action_dim)
        if self.cfg.mix_ratio > 0:
            perm = torch.randperm(B*T)
            with torch.no_grad():
                if self.cfg.feature_type == 'state':
                    desired_obs = processed_next_state[perm]
                    phi = self.feature_learner.feature_net(desired_obs)
                elif self.cfg.feature_type == 'diff':
                    desired_obs = processed_state[perm]
                    desired_next_obs = processed_next_state[perm]
                    phi = self.feature_learner.feature_net(desired_next_obs) - self.feature_learner.feature_net(desired_obs)
                else:
                    desired_obs = processed_state[perm]
                    desired_next_obs = processed_next_state[perm]
                    phi = torch.cat([self.feature_learner.feature_net(desired_obs), self.feature_learner.feature_net(desired_next_obs)], dim=-1)
            # compute inverse of cov of phi
            cov = torch.matmul(phi.T, phi) / phi.shape[0]
            inv_cov = torch.linalg.pinv(cov)

            mix_idxs: tp.Any = np.where(np.random.uniform(size=B*T) < self.cfg.mix_ratio)[0]
            with torch.no_grad():
                new_z = phi[mix_idxs]

            new_z = torch.matmul(new_z, inv_cov)  # batch_size x z_dim
            new_z = math.sqrt(self.cfg.z_dim) * F.normalize(new_z, dim=1)
            z[mix_idxs] = new_z

        critic_info = self.update_sf(obs=processed_state, action=processed_action, discount=self.cfg.hilp_discount, 
                                     next_obs=processed_next_state, future_obs=processed_future_state, z=z, step=self.steps)

        # update actor
        actor_info = self.update_actor(processed_state.detach(), z, self.steps)

        metrics.update(critic_info)
        metrics.update(actor_info)
        # update critic target
        utilss.soft_update_params(self.successor_net, self.successor_target_net, self.cfg.sf_target_tau)

        return metrics
    def state_dict(self):
        state = {
            # 'state_processor': self.state_processor.state_dict(),
            'actor': self.actor.state_dict(),
            'successor_net': self.successor_net.state_dict(),
            'successor_target_net': self.successor_target_net.state_dict(),
            'feature_learner': self.feature_learner.state_dict(),
        }
        return state
    def load_state_dict(self, state_dict):
        self.state_processor.load_state_dict(state_dict['state_processor'])
        self.actor.load_state_dict(state_dict['actor'])
        self.successor_net.load_state_dict(state_dict['successor_net'])
        self.successor_target_net.load_state_dict(state_dict['successor_target_net'])
        self.feature_learner.load_state_dict(state_dict['feature_learner'])
    def n_params(self):
        modules = [self.actor, self.successor_net, self.feature_learner]
        for m in modules:
            for n, p in m.named_parameters():
                if p.requires_grad:
                    print(f"{n}: {p.numel()}")
        n = sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)
        return n



class NewHILP(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, cfg) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.z_dim = z_dim
        self.cfg = cfg

        if self.cfg.feature_type != 'concat':
            feature_dim = z_dim
        else:
            assert z_dim % 2 == 0
            feature_dim = z_dim // 2

        self.phi1 = MLP(input_dim=obs_dim, hidden_dim=hidden_dim, output_dim=z_dim, hidden_depth=2)
        self.phi2 = MLP(input_dim=obs_dim, hidden_dim=hidden_dim, output_dim=z_dim, hidden_depth=2)
        self.target_phi1 = MLP(input_dim=obs_dim, hidden_dim=hidden_dim, output_dim=z_dim, hidden_depth=2)
        self.target_phi2 = MLP(input_dim=obs_dim, hidden_dim=hidden_dim, output_dim=z_dim, hidden_depth=2)
        self.target_phi1.load_state_dict(self.phi1.state_dict())
        self.target_phi2.load_state_dict(self.phi2.state_dict())

        self.apply(utilss.weight_init)

        # Define a running mean and std
        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_std', torch.ones(feature_dim))

    def feature_net(self, obs):
        phi = self.phi1(obs)
        phi = phi - self.running_mean
        return phi

    def value(self, obs: torch.Tensor, goals: torch.Tensor, is_target: bool = False):
        if is_target:
            phi1 = self.target_phi1
            phi2 = self.target_phi2
        else:
            phi1 = self.phi1
            phi2 = self.phi2

        phi1_s = phi1(obs)
        phi1_g = phi1(goals)

        phi2_s = phi2(obs)
        phi2_g = phi2(goals)

        squared_dist1 = ((phi1_s - phi1_g) ** 2).sum(dim=-1)
        v1 = -torch.sqrt(torch.clamp(squared_dist1, min=1e-6))
        squared_dist2 = ((phi2_s - phi2_g) ** 2).sum(dim=-1)
        v2 = -torch.sqrt(torch.clamp(squared_dist2, min=1e-6))

        if is_target:
            v1 = v1.detach()
            v2 = v2.detach()

        return v1, v2

    def expectile_loss(self, adv, diff, expectile=0.7):
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        goals = future_obs
        rewards = (torch.linalg.norm(obs - goals, dim=-1) < 1e-6).float()
        masks = 1.0 - rewards
        rewards = rewards - 1.0

        next_v1, next_v2 = self.value(next_obs, goals, is_target=True)
        next_v = torch.minimum(next_v1, next_v2)
        q = rewards + self.cfg.hilp_discount * masks * next_v

        v1_t, v2_t = self.value(obs, goals, is_target=True)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = rewards + self.cfg.hilp_discount * masks * next_v1
        q2 = rewards + self.cfg.hilp_discount * masks * next_v2
        v1, v2 = self.value(obs, goals, is_target=False)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.cfg.hilp_expectile).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.cfg.hilp_expectile).mean()
        value_loss = value_loss1 + value_loss2

        utilss.soft_update_params(self.phi1, self.target_phi1, 0.005)
        utilss.soft_update_params(self.phi2, self.target_phi2, 0.005)

        with torch.no_grad():
            phi1 = self.phi1(obs)
            self.running_mean = 0.995 * self.running_mean + 0.005 * phi1.mean(dim=0)
            self.running_std = 0.995 * self.running_std + 0.005 * phi1.std(dim=0)

        return value_loss, {
            'hilp/value_loss': value_loss,
            'hilp/v_mean': v.mean(),
            'hilp/v_max': v.max(),
            'hilp/v_min': v.min(),
            'hilp/abs_adv_mean': torch.abs(adv).mean(),
            'hilp/adv_mean': adv.mean(),
            'hilp/adv_max': adv.max(),
            'hilp/adv_min': adv.min(),
            'hilp/accept_prob': (adv >= 0).float().mean(),
        }