import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal, SigmoidTransform, AffineTransform, TransformedDistribution
from torch import distributions as pyd
import os

# from utils.util import unpack_batch, RunningMeanStd
from utils.util import unpack_batch
from utils.util import MLP

from agent.sac.sac_agent import SACAgent, DoubleQCritic
from agent.sac.actor import DiagGaussianActor
from torchinfo import summary
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class truncated_normal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, mean, std, low, high):
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high
        self.normal_dist = Normal(mean, std)
        # Transform the standard normal into a truncated range
        # SigmoidTransform maps (-inf, inf) -> (0, 1)
        # AffineTransform scales (0, 1) -> (low, high)
        self.trunc_transform = torch.distributions.transforms.ComposeTransform([
            SigmoidTransform(),  # Maps to (0, 1)
            AffineTransform(loc=low, scale=high - low)  # Maps (0, 1) -> (low, high)
        ])
        super().__init__(self.normal_dist, self.trunc_transform)


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


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth)

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
        self.trunk1 = mlp(input_dim, hidden_dim, output_dim, hidden_depth)
        self.trunk2 = mlp(input_dim, hidden_dim, output_dim, hidden_depth)

    def forward(self, x):
        return self.trunk1(x), self.trunk2(x)

def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


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


class SPEDERSACAgent():
    """
    SAC with VAE learned latent features
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space,
            phi_and_mu_lr=-1,
            # 3e-4 was originally proposed in the paper, but seems to results in fluctuating performance
            phi_hidden_dim=-1,
            phi_hidden_depth=-1,
            mu_hidden_dim=-1,
            mu_hidden_depth=-1,
            critic_and_actor_lr=-1,
            critic_and_actor_hidden_dim=-1,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=1024,
            feature_tau=0.005,
            feature_dim=2048,  # latent feature dim
            use_feature_target=True,
            extra_feature_steps=1,
            device='cuda:0',
            state_task_dataset=None,
            lasso_coef=1e-3,
            n_task=3,
            learnable_temperature=False,
    ):

        # super().__init__(
        #     state_dim=state_dim,
        #     action_dim=action_dim,
        #     action_space=action_space,
        #     tau=tau,
        #     alpha=alpha,
        #     discount=discount,
        #     target_update_period=target_update_period,
        #     auto_entropy_tuning=auto_entropy_tuning,
        #     hidden_dim=hidden_dim,
        #     device=device
        # )

        # state_dataset = state_task_dataset[:, :state_dim]
        # mean, std = state_dataset.mean(0), state_dataset.std(0)
        # low, high = state_dataset.min(0)[0], state_dataset.max(0)
        # self.low, self.high = low, high
        # self.obs_dist = pyd.Uniform(low=torch.FloatTensor(low).to(device), high=torch.FloatTensor(high).to(device))

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.feature_tau = feature_tau
        self.use_feature_target = use_feature_target
        self.extra_feature_steps = extra_feature_steps
        self.discount = discount
        self.device = device
        self.log_alpha = torch.tensor(np.log(alpha)).to(self.device)
        self.log_alpha.requires_grad = True
        self.steps = 0
        self.lasso_coef = lasso_coef
        self.n_task = n_task
        self.phi_hidden_dim = phi_hidden_dim
        self.learnable_temperature = learnable_temperature
        self.target_update_period = target_update_period
        self.tau = tau
        self.normalize_dict = torch.load(f'./kms/normalize_dict.pth')
        # for key, value in self.normalize_dict.items():
        #     self.normalize_dict[key] = torch.FloatTensor(value).to(self.device)
        self.phi = MLP(input_dim=state_dim + action_dim,
                       output_dim=feature_dim,
                       hidden_dim=phi_hidden_dim,
                       hidden_depth=phi_hidden_depth).to(device)

        if use_feature_target:
            self.phi_target = copy.deepcopy(self.phi)

        self.mu = MLP(input_dim=state_dim,
                      output_dim=feature_dim,
                      hidden_dim=mu_hidden_dim,
                      hidden_depth=mu_hidden_depth).to(device)

        # self.theta = Theta(feature_dim=feature_dim).to(device)

        self.feature_optimizer = torch.optim.Adam(
            list(self.phi.parameters()) + list(self.mu.parameters()),
            weight_decay=0, lr=phi_and_mu_lr)
        self.actor = DiagGaussianActor(
            obs_dim=state_dim+n_task,
            action_dim=action_dim,
            hidden_dim=critic_and_actor_hidden_dim,
            hidden_depth=2,
            log_std_bounds=[-5., 2.],
        ).to(device)
        # self.critic = RFFCritic(feature_dim=feature_dim, hidden_dim=critic_and_actor_hidden_dim).to(device)
        self.critic = DoubleMLP(input_dim=self.n_task,
                          output_dim=feature_dim,
                          hidden_dim=critic_and_actor_hidden_dim,
                            hidden_depth=1).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.w = MLP(input_dim=self.n_task,
                     output_dim=feature_dim,
                     hidden_dim=critic_and_actor_hidden_dim,
                     hidden_depth=1).to(device)
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()),
                                                weight_decay=0, lr=critic_and_actor_lr,
                                                betas=[0.9, 0.999])  # lower lr for actor/alpha
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=critic_and_actor_lr, betas=[0.9, 0.999])


        self.critic_optimizer = torch.optim.Adam(list(self.critic.parameters())+list(self.w.parameters())+list(self.phi.parameters()),
                                                 weight_decay=0, lr=critic_and_actor_lr, betas=[0.9, 0.999])

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def update_target(self):
        if self.steps % self.target_update_period == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def getQ_detach_phi(self, state, action, task_onehot):
        z_phi = self.phi(torch.concat([state, action], -1)).detach()
        u1, u2 = self.critic(task_onehot)
        q1 = torch.sum(z_phi * u1, dim=-1, keepdim=True)
        q2 = torch.sum(z_phi * u2, dim=-1, keepdim=True)
        q = torch.min(q1, q2)
        return q
    def get_targetQ(self, state, action, task_onehot):
        z_phi = self.phi(torch.concat([state, action], -1))
        target_u1, target_u2 = self.critic_target(task_onehot)
        target_q1 = torch.sum(z_phi * target_u1, dim=-1, keepdim=True)
        target_q2 = torch.sum(z_phi * target_u2, dim=-1, keepdim=True)
        target_q = torch.min(target_q1, target_q2)
        return target_q
    
    def getV(self, state, task_onehot):
        dist = self.actor(torch.cat([state, task_onehot], -1))
        action = dist.rsample()
        q = self.getQ_detach_phi(state, action, task_onehot)
        v = q - self.alpha.detach() * dist.log_prob(action).sum(-1, keepdim=True)
        return v
    
    def get_targetV(self, state, task_onehot):
        dist = self.actor(torch.cat([state, task_onehot], -1))
        action = dist.sample()
        target_q = self.get_targetQ(state, action, task_onehot)
        target_v = target_q - self.alpha.detach() * dist.log_prob(action).sum(-1, keepdim=True)
        return target_v

    def feature_step(self, batch, s_random, a_random, s_prime_random):
        """
        Loss implementation
        """

        state, action, next_state, reward, _, task, next_task = unpack_batch(batch)

        z_phi = self.phi(torch.concat([state, action], -1))
        z_phi_random = self.phi(torch.concat([s_random, a_random], -1))

        z_mu_next = self.mu(next_state)
        z_mu_next_random = self.mu(s_prime_random)

        assert z_phi.shape[-1] == self.feature_dim
        assert z_mu_next.shape[-1] == self.feature_dim

        model_loss_pt1 = -2 * torch.diag(z_phi @ z_mu_next.T)  # check if need to sum

        model_loss_pt2_a = z_phi_random @ z_mu_next.T
        model_loss_pt2 = model_loss_pt2_a @ model_loss_pt2_a.T

        model_loss_pt1_summed = 1. / torch.numel(model_loss_pt1) * torch.sum(model_loss_pt1)
        model_loss_pt2_summed = 1. / torch.numel(model_loss_pt2) * torch.sum(model_loss_pt2)

        model_loss = model_loss_pt1_summed + model_loss_pt2_summed


        W = self.phi.trunk[0].weight
        assert W.shape == (self.phi_hidden_dim, self.state_dim + self.action_dim)
        group_by_coordinate_W = W.reshape(self.phi_hidden_dim, (self.state_dim + self.action_dim)//2, 2)
        group_lasso = torch.sqrt(group_by_coordinate_W.pow(2).sum(-1).sum(0)).sum()
        # print('W', W.shape)

        loss = model_loss + group_lasso * self.lasso_coef
        # print('model_loss', model_loss)
        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()

        return {
            'total_loss': loss.item(),
            'model_loss': model_loss.item(),
            'group_lasso': group_lasso.item(),
            # 'prob_loss': prob_loss.item(),
        }

    def update_feature_target(self):
        for param, target_param in zip(self.phi.parameters(), self.phi_target.parameters()):
            target_param.data.copy_(self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)


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
        """
        Critic update step
        """
        expert_state, expert_action, expert_next_state, expert_reward, expert_done, expert_task, expert_next_task = unpack_batch(batch)
        assert expert_state.shape[-1] == self.state_dim
        assert expert_action.shape[-1] == self.action_dim
        assert expert_next_state.shape[-1] == self.state_dim
        assert expert_done.shape[-1] == 1
        expert_task_onehot = torch.eye(self.n_task)[expert_task.long().reshape(-1)].to(self.device)
        # print('task_onehot', task_onehot.shape)
        assert expert_task_onehot.shape == (expert_state.shape[0], self.n_task)
        expert_next_task_onehot = torch.eye(self.n_task)[expert_next_task.long().reshape(-1)].to(self.device)
        assert expert_next_task_onehot.shape == (expert_state.shape[0], self.n_task)
        # calculate r
        z_phi = self.phi(torch.concat([expert_state, expert_action], -1)) # only need gradient in this place
        w_batch = self.w(expert_task_onehot)
        r = torch.sum(z_phi * w_batch, dim=-1, keepdim=True)
        # calculate target
        target_q = r + (1 - expert_done) * self.discount * self.get_targetV(expert_next_state, expert_next_task_onehot).detach()

        u1, u2 = self.critic(expert_task_onehot)
        q1 = torch.sum(z_phi * u1, dim=-1, keepdim=True).detach()
        q2 = torch.sum(z_phi * u2, dim=-1, keepdim=True).detach()
        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        qr_loss = q1_loss + q2_loss

        ##IQ Learn loss
        iq_alpha = 0.5
        
        dist = self.actor(torch.cat([expert_next_state, expert_next_task_onehot], -1))
        sample_next_action = dist.sample().detach() # irrelevant to actor
        next_Q = self.get_targetQ(expert_next_state, sample_next_action, expert_next_task_onehot).detach()
        sample_next_action_logprob = dist.log_prob(sample_next_action)
        next_V = (next_Q - self.alpha.detach() * sample_next_action_logprob.sum(-1, keepdim=True))
        # -E_(ρ_expert)[Q(s, a) - γV(s')]
        y = (1 - expert_done) * self.discount * next_V.detach()
        # print('next_Q:', torch.where(torch.isnan(next_Q.flatten())))
        # print('logprob:', torch.where(torch.isnan(dist.log_prob(sample_next_action))))
        # if torch.isnan(sample_next_action_logprob).any():
        #     print('state:', expert_next_state[187], torch.where(torch.isnan(expert_next_state)))
        #     print('action:', sample_next_action[187,47], torch.where(torch.isnan(sample_next_action)))
        #     print('task:', expert_next_task.flatten()[187], torch.where(torch.isnan(expert_next_task)))
        expert_Q = self.getQ_detach_phi(expert_state, expert_action, expert_task_onehot)
        r = expert_Q - y # the gradient is only with expert Q
        loss_1 = -r.mean()

        # E_(ρ)[V(s) - γV(s')]
        dist = self.actor(torch.cat([expert_state, expert_task_onehot], -1))
        sample_action = dist.sample().detach() # irrelevant to actor
        current_Q = self.getQ_detach_phi(expert_state, sample_action, expert_task_onehot)
        action_logprob = dist.log_prob(sample_action).sum(-1, keepdim=True)
        current_V = current_Q - self.alpha.detach() * action_logprob # the gradient is only with sample_Q and V
        loss_2 = (current_V - y).mean()

        # regularization
        loss_3 = 1/(4*iq_alpha) * (r**2).mean()

        iql_loss = loss_1 + loss_2 + loss_3
        # print('loss_1:', loss_1.item())
        # print('current_Q:', current_Q.flatten(), 'action_prob:', action_logprob)
        # print('isnan:', torch.where(torch.isnan(action_logprob)), torch.where(torch.isnan(current_Q.flatten())), 
        #       'y:', torch.where(torch.isnan(y)))
        # print('loss_2:', loss_2.item())
        # print('loss_3:', loss_3.item())
        ##

        u1_l1_loss = torch.abs(u1).mean()
        u2_l1_loss = torch.abs(u2).mean()
        w_l1_loss = torch.abs(w_batch).mean(-1).norm(2,dim=0)

        l1_loss = (u1_l1_loss + u2_l1_loss + w_l1_loss) * 0.2

        #
        loss = qr_loss + l1_loss
        # print('q_loss', q_loss)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item(),
            'l1_loss': l1_loss.item(),
            'w_l1_loss': w_l1_loss.item(),
            'u1_l1_loss': u1_l1_loss.item(),
            'iql_loss_1': loss_1.item(),
            'iql_loss_2': loss_2.item(),
            'iql_loss_3': loss_3.item(),
            'iql_loss': iql_loss.item(),
        }

    def another_critic_step(self, batch):       
        """
        Critic update step
        """
        state, action, next_state, reward, done, task, next_task = unpack_batch(batch)
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim
        assert done.shape[-1] == 1
        task_onehot = torch.eye(self.n_task)[task.long().reshape(-1)].to(self.device)
        assert task_onehot.shape == (state.shape[0], self.n_task)
        next_task_onehot = torch.eye(self.n_task)[next_task.long().reshape(-1)].to(self.device)
        assert next_task_onehot.shape == (state.shape[0], self.n_task)

        # critic step, iql loss
        z_phi = self.phi(torch.concat([state, action], -1))
        u1, u2 = self.critic(task_onehot)
        q1 = torch.sum(z_phi * u1, dim=-1, keepdim=True)
        q2 = torch.sum(z_phi * u2, dim=-1, keepdim=True)
        next_v = self.get_targetV(next_state, next_task_onehot).detach()
        current_v = self.getV(state, task_onehot)
        q = torch.min(q1, q2)
        # iq_loss_1 = self.iq_loss(q1, current_v, next_v, done)
        # iq_loss_2 = self.iq_loss(q2, current_v, next_v, done)
        iq_alpha = 0.5
        y = (1 - done) * self.discount * next_v
        r = q - y
        loss_1 = -r.mean()
        loss_2 = (current_v - y).mean()
        loss_3 = 1/(4*iq_alpha) * (r**2).mean()
        # iq_loss = (iq_loss_1 + iq_loss_2) / 2

        # r step, w update
        w_batch = self.w(task_onehot)
        r_phiw = torch.sum(z_phi * w_batch, dim=-1, keepdim=True)
        target_q = self.get_targetQ(state, action, task_onehot)
        target_r = target_q.detach() - (1 - done) * self.discount * next_v
        r_loss = F.mse_loss(r_phiw, target_r)
        
        # l1 loss
        u1_l1_loss = torch.abs(u1).mean()
        u2_l1_loss = torch.abs(u2).mean()
        w_l1_loss = torch.abs(w_batch).mean()
        l1_loss = (u1_l1_loss + u2_l1_loss) / 3

        # update
        loss = iq_loss + r_loss
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            # 'iq_loss_1': iq_loss_1.item(),
            # 'iq_loss_2': iq_loss_2.item(),
            'iq_loss': iq_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item(),
            'l1_loss': l1_loss.item(),
            # 'w_l1_loss': w_l1_loss.item(),
            'u1_l1_loss': u1_l1_loss.item(),
        }
    def update_actor_and_alpha(self, batch):
        """
        Actor update step
        """
        expert_state, expert_action, expert_next_state, expert_reward, expert_done, expert_task, expert_next_task = unpack_batch(batch)
        assert expert_state.shape[-1] == self.state_dim
        assert expert_action.shape[-1] == self.action_dim
        assert expert_next_state.shape[-1] == self.state_dim
        assert expert_done.shape[-1] == 1
        expert_task_onehot = torch.eye(self.n_task)[expert_task.long().reshape(-1)].to(self.device)
        expert_next_task_onehot = torch.eye(self.n_task)[expert_next_task.long().reshape(-1)].to(self.device)
        expert_state_task = torch.cat([expert_state, expert_task_onehot], -1)



        dist = self.actor(expert_state_task)
        expert_log_prob = dist.log_prob(expert_action).sum(-1, keepdim=True)
        negll = -expert_log_prob.mean()

        sample_action = dist.rsample()
        sample_log_prob = dist.log_prob(sample_action).sum(-1, keepdim=True)

        z_phi_sample = self.phi(torch.concat([expert_state, sample_action], -1))

        u1, u2 = self.critic(expert_task_onehot) # irrelevant to critic
        sample_q1 = torch.sum(z_phi_sample * u1, dim=-1, keepdim=True)
        sample_q2 = torch.sum(z_phi_sample * u2, dim=-1, keepdim=True)
        sample_q = torch.min(sample_q1, sample_q2)

        SAC_loss = ((self.alpha) * sample_log_prob - sample_q).mean()

        actor_loss = SAC_loss + negll
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        info = {'actor_loss': actor_loss.item(),
                'negll': negll.item(),
                'SAC_loss': SAC_loss.item()}

        # print('actor_loss', actor_loss)
        # print('negll', negll)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-sample_log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            info['alpha_loss'] = alpha_loss
            info['alpha'] = self.alpha

        return info


    def state_dict(self):
        return {'actor': self.actor.state_dict(),
				'critic': self.critic.state_dict(),
				'log_alpha': self.log_alpha,
				'phi': self.phi.state_dict(),
				'mu': self.mu.state_dict(),
                'w': self.w.state_dict()}
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.log_alpha = state_dict['log_alpha']
        self.phi.load_state_dict(state_dict['phi'])
        self.mu.load_state_dict(state_dict['mu'])
        self.w.load_state_dict(state_dict['w'])
        # self.theta.load_state_dict(state_dict['theta'])

    def load_phi_mu(self, state_dict):
        self.phi.load_state_dict(state_dict['phi'])
        self.mu.load_state_dict(state_dict['mu'])

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        # Feature step
        for _ in range(self.extra_feature_steps + 1):
            batch_1 = buffer.sample(batch_size)
            batch_2 = buffer.sample(batch_size)
            s_random, a_random, s_prime_random, _, _, task_random, next_task_random = unpack_batch(batch_2)
            # s_random = st_random[:, :self.state_dim]
            # s_prime_random = self.obs_dist.sample((batch_size,)).to(self.device)
            feature_info = self.feature_step(batch_1, s_random, a_random, s_prime_random)

            # Update the feature network if needed
            if self.use_feature_target:
                self.update_feature_target()
        batch_1 = buffer.sample(batch_size)
        # Critic step, IRL
        critic_info = self.critic_step(batch_1)

        # Actor and alpha step, make the actor closer to softmaxQ
        actor_info = self.update_actor_and_alpha(batch_1)

        # Update the frozen target models
        self.update_target()

        return {
            **feature_info,
            **critic_info,
            **actor_info,
        }
    
    def log_likelihood(self, state, action, next_state, kde=False):
        # output the device
        with torch.no_grad():
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            z_phi = self.phi(torch.concat([state, action], -1))
            z_mu_next = self.mu(next_state)
            # next_state_p0 = self.obs_dist.log_prob(next_state).sum(-1, keepdim=True).exp()
            # print('next_state_p0', next_state_p0.mean(0))
            # if kde == False:
            #     next_state_gaussian_aux = 1/torch.sqrt(2*torch.tensor([torch.pi])) * torch.exp(-0.5*next_state**2)
            #     next_state_gaussian = torch.prod(next_state_gaussian_aux, dim=-1)  
            # else:
            #     pass #TODO
            # print(z_phi)
            # print(z_mu_next)
            # print(next_state)
            # print(next_state_gaussian)
            # print(torch.sum(z_phi*z_mu_next, dim=-1))
            likelihood = torch.sum(z_phi*z_mu_next, dim=-1)
            # print(likelihood)
        return likelihood.mean()
    
    def action_loglikelihood(self, state, action, task):
        task_onehot = torch.eye(self.n_task)[task.long().reshape(-1)].to(self.device)
        state_task = torch.cat([state, task_onehot], -1)

        dist = self.actor(state_task)
        actor_log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        z_phi = self.phi(torch.concat([state, action], -1))
        u1, u2 = self.critic(task_onehot)
        q1 = torch.sum(z_phi * u1, dim=-1, keepdim=True)
        q2 = torch.sum(z_phi * u2, dim=-1, keepdim=True)
        q = torch.min(q1, q2)

        return actor_log_prob.mean()
    
    def generate_next_state(self, state, action):
        state_max, state_min = self.normalize_dict['state_max'], self.normalize_dict['state_min']
        action_max, action_min = self.normalize_dict['action_max'], self.normalize_dict['action_min']
        if 'next_state_max' in self.normalize_dict:
            next_state_max, next_state_min = self.normalize_dict['next_state_max'], self.normalize_dict['next_state_min']
        else:
            next_state_max, next_state_min = state_max, state_min
        scale_back_state = (state_max - state_min) * state + state_min
        scale_back_action = (action_max - action_min) * action + action_min
        predict_next_state = scale_back_state + scale_back_action
        scaled_next_state = (predict_next_state - next_state_min) / (next_state_max - next_state_min)
        return scaled_next_state
    
    def step(self, state, task, action):
        with torch.no_grad():
            next_state = self.generate_next_state(state, action)
            task_onehot = torch.eye(self.n_task)[task].to(self.device)
            dist = self.actor(torch.cat([next_state, task_onehot], -1))
            next_action = dist.sample()
            z_phi = self.phi(torch.concat([state, action], -1))
            mu_next = self.mu(next_state)
            sp_likelihood = torch.sum(z_phi * mu_next, dim=-1)
            u1, u2 = self.critic(task_onehot)
            q1 = torch.sum(z_phi * u1, dim=-1, keepdim=True)
            q2 = torch.sum(z_phi * u2, dim=-1, keepdim=True)
            q = torch.min(q1, q2)
            unnormlized_action_logprob = q
        return next_state, next_action, sp_likelihood, unnormlized_action_logprob
            



