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
from agent.sac.actor import DiagGaussianActor, MultiSoftmaxActor
from torchinfo import summary
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SPEDERSACAgent():
    """
    SAC with VAE learned latent features
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space=None,
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
            state_dataset=None,
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
        self.action_dim = action_dim * 5
        self.n_action = 5
        self.n_action_dim = self.action_dim // self.n_action
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
        self.phi = Norm1MLP_singlelayer(input_dim=state_dim + self.action_dim,
                       output_dim=feature_dim).to(device)
        # self.phi = Norm1MLP(input_dim=state_dim + self.action_dim,
        #                     hidden_dim=phi_hidden_dim,
        #                     output_dim=feature_dim).to(device)
        # self.phi = MLP(input_dim=state_dim + self.action_dim,
        #                 output_dim=feature_dim,
        #                 hidden_dim=phi_hidden_dim,
        #                 hidden_depth=1).to(device)
        # self.phi = RFFMLP(input_dim=state_dim + action_dim,
        #                   hidden_dim=state_dim,
        #                   output_dim=feature_dim).to(device)
        if use_feature_target:
            self.phi_target = copy.deepcopy(self.phi)
        # self.mu = Norm1MLP(input_dim=state_dim,
        #                 output_dim=feature_dim,
        #                 hidden_dim=mu_hidden_dim).to(device)
        # self.mu = MLP(input_dim=state_dim,
        #                 output_dim=feature_dim,
        #                 hidden_dim=mu_hidden_dim,
        #                 hidden_depth=1).to(device)
        self.mu = Norm1MLP(input_dim=state_dim,
                           hidden_dim=mu_hidden_dim,
                           output_dim=feature_dim).to(device)
        # self.mu = RFFMLP(input_dim=state_dim,
                        #  hidden_dim=state_dim,
                    #   output_dim=feature_dim).to(device)
        # self.mu.l1.weight.data = copy.deepcopy(self.phi.l1.weight.data)
        # self.mu.l1.bias.data = copy.deepcopy(self.phi.l1.bias.data)
        # print('l0 phi grad:', self.phi.l0.weight.requires_grad, self.phi.l0.bias.requires_grad)
        # print('l0 mu grad:', self.mu.l0.weight.requires_grad, self.mu.l0.bias.requires_grad)
        # print('phi grad:', self.phi.l1.weight.requires_grad, self.phi.l1.bias.requires_grad)
        # print('mu grad:', self.mu.l1.weight.requires_grad, self.mu.l1.bias.requires_grad)
        # assert torch.isclose(self.phi.l1.weight, self.mu.l1.weight).all()
        # assert torch.isclose(self.phi.l1.bias, self.mu.l1.bias).all()
        # self.theta = Theta(feature_dim=feature_dim).to(device)

        self.feature_optimizer = torch.optim.Adam(
            list(self.phi.parameters()) + list(self.mu.parameters()),
            weight_decay=0, lr=phi_and_mu_lr)
        # self.actor = DiagGaussianActor(
        #     obs_dim=state_dim+n_task,
        #     action_dim=action_dim,
        #     hidden_dim=critic_and_actor_hidden_dim,
        #     hidden_depth=2,
        #     log_std_bounds=[-8., 2.],
        # ).to(device)
        self.actor = MultiSoftmaxActor(
            obs_dim=state_dim+n_task,
            action_dim=self.action_dim,
            n_action=self.n_action,
            hidden_dim=critic_and_actor_hidden_dim,
            hidden_depth=2,
        ).to(device)
        # self.critic = RFFCritic(feature_dim=feature_dim+n_task, hidden_dim=critic_and_actor_hidden_dim).to(device)
        # self.critic = DoubleMLP(inptu_dim=self.n_task,
        #                   output_dim=feature_dim,
        #                   hidden_dim=critic_and_actor_hidden_dim,
        #                     hidden_depth=1).to(device)
        # self.critic = RFF_complex_critic(feature_dim=state_dim+n_task, hidden_dim=critic_and_actor_hidden_dim).to(device)
        # self.u = MLP(input_dim=self.n_task,
        #                 output_dim=feature_dim,
        #                 hidden_dim=critic_and_actor_hidden_dim).to(device)
        # self.u_target = copy.deepcopy(self.u)
        # self.critic = DoubleMLP(input_dim=n_task,
        #                   output_dim=feature_dim,
        #                   hidden_dim=critic_and_actor_hidden_dim,
        #                   hidden_depth=1).to(device)
        # self.critic = MLP(input_dim=feature_dim,
        #                             output_dim=feature_dim,
        #                             hidden_dim=critic_and_actor_hidden_dim,
        #                             hidden_depth=1).to(device)
        self.critic = Norm1MLP(input_dim=feature_dim,
                                output_dim=feature_dim,
                                hidden_dim=critic_and_actor_hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        # self.u = MLP(input_dim=n_task,
        #              output_dim=feature_dim,
        #              hidden_dim=critic_and_actor_hidden_dim,
        #              hidden_depth=0).to(device)
        self.u = Norm1MLP(input_dim=self.n_task,
                    output_dim=feature_dim,
                    hidden_dim=critic_and_actor_hidden_dim).to(device)
        self.w = MLP(input_dim=self.n_task,
                     output_dim=feature_dim,
                     hidden_dim=critic_and_actor_hidden_dim,
                     hidden_depth=0).to(device)
        # self.beta = torch.FloatTensor([1]).to(device)
        # self.beta.requires_grad = True
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()),
                                                weight_decay=0, lr=critic_and_actor_lr,
                                                betas=[0.9, 0.999])  # lower lr for actor/alpha
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=critic_and_actor_lr, betas=[0.9, 0.999])

        self.critic_optimizer = torch.optim.Adam(list(self.critic.parameters())+list(self.u.parameters())+list(self.w.parameters()),
                                                    weight_decay=0, lr=critic_and_actor_lr, betas=[0.9, 0.999])
        # self.u_optimizer = torch.optim.Adam(list(self.u.parameters())+list(self.w.parameters()),
        #                                          weight_decay=0, lr=critic_and_actor_lr, betas=[0.9, 0.999])
        # self.beta_optimizer = torch.optim.Adam([self.beta], lr=1e-2, betas=[0.9, 0.999])
    
    def rescale_state_back(self, state):
        return state * self.state_std + self.state_mean
    def rescale_action_back(self, action):
        return action * self.action_std + self.action_mean
    def rescalse_state(self, state):
        return (state - self.state_mean) / self.state_std
    def rescale_action(self, action):
        return (action - self.action_mean) / self.action_std
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
    def get_targetQ_u(self, state, action, task_onehot):
        z_phi = self.phi(torch.concat([state, action], -1))
        f_phi = self.critic_target(z_phi)
        target_u = self.u(task_onehot)
        target_q = torch.sum(f_phi * target_u, dim=-1, keepdim=True)
        return target_q
    def get_targetQ_QR(self, state, action, task_onehot):
        z_phi = self.phi(torch.concat([state, action], -1))
        target_qr = self.critic_target(z_phi)
        target_q = target_qr[:, 0:1]
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
        action_onehot = torch.eye(self.n_action)[action.long()].reshape(-1, self.action_dim).to(self.device)
        target_q = self.get_targetQ_u(state, action_onehot, task_onehot)
        target_v = target_q - self.alpha.detach() * dist.log_prob(action).sum(-1, keepdim=True)
        return target_v

    def feature_step_discrete(self, batch, s_random, a_random, s_prime_random):
        """
        Loss implementation
        """

        state, action, next_state, reward, _, task, next_task = unpack_batch(batch)
        action_onehot = torch.eye(self.n_action)[action.long()].reshape(-1, self.action_dim).to(self.device)
        z_phi = self.phi(torch.concat([state, action_onehot], -1))
        # z_phi = self.phi(torch.concat([state, action], -1))

        # z_phi_random = self.phi(torch.concat([s_random, a_random], -1))

        z_mu_next = self.mu(next_state)
        z_mu_next_random = self.mu(s_prime_random)

        assert z_phi.shape[-1] == self.feature_dim
        assert z_mu_next.shape[-1] == self.feature_dim

        model_loss_pt1 = -2 * torch.diag(z_phi @ z_mu_next.T)  # check if need to sum

        model_loss_pt2_a = z_phi @ z_mu_next_random.T
        model_loss_pt2 = model_loss_pt2_a @ model_loss_pt2_a.T

        model_loss_pt1_summed = 1. / torch.numel(model_loss_pt1) * torch.sum(model_loss_pt1)
        model_loss_pt2_summed = 1. / torch.numel(model_loss_pt2) * torch.sum(model_loss_pt2)

        model_loss = model_loss_pt1_summed + model_loss_pt2_summed
        loss = model_loss
        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()
        return {
            'total_loss': loss.item(),
            'model_loss': model_loss.item(),
            # 'prob_loss': prob_loss.item(),
            # 'positive_ll': positive_loss.item(),
            # 'negative_ll_1': negative_loss_1.item(),
            # 'negative_ll_2': negative_loss_2.item(),
        }

    def feature_step(self, batch, s_random, a_random, s_prime_random):
        """
        Loss implementation
        """

        state, action, next_state, reward, _, task, next_task = unpack_batch(batch)
        action_onehot = torch.eye(self.n_action)[action.long()].reshape(-1, self.action_dim).to(self.device)
        z_phi = self.phi(torch.concat([state, action_onehot], -1))
        # z_phi = self.phi(torch.concat([state, action], -1))

        # z_phi_random = self.phi(torch.concat([s_random, a_random], -1))

        z_mu_next = self.mu(next_state)
        # z_mu_next_random = self.mu(s_prime_random)

        assert z_phi.shape[-1] == self.feature_dim
        assert z_mu_next.shape[-1] == self.feature_dim

        # model_loss_pt1 = -2 * torch.diag(z_phi @ z_mu_next.T)  # check if need to sum

        # model_loss_pt2_a = z_phi_random @ z_mu_next_random.T
        # model_loss_pt2 = model_loss_pt2_a @ model_loss_pt2_a.T

        # model_loss_pt1_summed = 1. / torch.numel(model_loss_pt1) * torch.sum(model_loss_pt1)
        # model_loss_pt2_summed = 1. / torch.numel(model_loss_pt2) * torch.sum(model_loss_pt2)

        # model_loss = model_loss_pt1_summed + model_loss_pt2_summed


        W = self.phi.l1.weight
        group_by_coordinate_W = W.reshape(self.feature_dim, (self.state_dim + self.action_dim)//2, 2)
        group_lasso = torch.sqrt(group_by_coordinate_W.pow(2).mean(-1)).mean()
        W_l1 = group_lasso
        ll_ctrl = z_phi @ z_mu_next.T
        loss_ctrl = torch.nn.CrossEntropyLoss()(ll_ctrl, torch.eye(state.shape[0]).to(self.device))

        loss = loss_ctrl
        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()

        return {
            # 'total_loss': loss.item(),
            # 'model_loss': model_loss.item(),
            # 'group_lasso': group_lasso.item(),
            # 'prob_loss': prob_loss.item(),
            # 'positive_ll': positive_loss.item(),
            # 'negative_ll_1': negative_loss_1.item(),
            # 'negative_ll_2': negative_loss_2.item(),
            'loss_ctrl': loss_ctrl.item(),
            'loss_feature': loss.item(),
            'W_l1': W_l1.item(),
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

    def critic_step_discrete(self, batch, s_random, a_random, s_prime_random, task_random):
        """
        We can not do behavior cloning using a Q function parametrized by u, so it's better for us to use an actor first and 
            use u to approximate that actor
        """
        ##TODO: Add w to the optimizer
        expert_state, expert_action, expert_next_state, expert_reward, expert_done, expert_task, expert_next_task = unpack_batch(batch)
        # print('expert_task', expert_task)
        assert expert_state.shape[-1] == self.state_dim
        assert expert_action.shape[-1] == self.action_dim//self.n_action
        assert expert_next_state.shape[-1] == self.state_dim
        assert expert_done.shape[-1] == 1
        expert_task_onehot = torch.eye(self.n_task)[expert_task.long().squeeze(-1)].to(self.device)
        # print('task_onehot', task_onehot.shape)
        assert expert_task_onehot.shape == (expert_state.shape[0], self.n_task)
        batch_size = expert_state.shape[0]
        batch_state = expert_state.reshape(batch_size, 1, self.state_dim)
        batch_action = expert_action.reshape(1, batch_size, self.action_dim//self.n_action)
        batch_action_onehot = torch.eye(self.n_action)[batch_action.long()].reshape(1, batch_size, self.action_dim).to(self.device)
        batch_state_action = torch.concat([batch_state.expand(-1, batch_size, -1), batch_action_onehot.expand(batch_size, -1, -1)], dim=-1)
        batch_phi = self.phi(batch_state_action).detach()
        batch_f_phi = self.critic(batch_phi)
        batch_task_onehot = expert_task_onehot.reshape(batch_size, 1, self.n_task).expand(-1, batch_size, -1)
        batch_u = self.u(batch_task_onehot)
        assert batch_phi.shape == batch_u.shape
        q_data = torch.sum(batch_f_phi * batch_u, dim=-1, keepdim=False)
        # n_Gibbs = 10
        # q_all = torch.zeros(batch_size, batch_size).to(self.device)
        # q_all[:, :batch_size] = q_data
        q_all = q_data
        # write a parallel version
        # batch_action_Gibbs = expert_action.reshape(batch_size, 1, self.n_action_dim).expand(-1, n_Gibbs, -1)
        # batch_state_Gibbs = expert_state.reshape(batch_size, 1, self.state_dim).expand(-1, n_Gibbs, -1)
        # batch_task_Gibbs = expert_task.reshape(batch_size, 1, 1).expand(-1, n_Gibbs, -1)
        # batch_action_prime = self.Gibbs_step(batch_state_Gibbs.reshape(-1, self.state_dim), \
        #                                      batch_action_Gibbs.reshape(-1, self.n_action_dim), \
        #                                         batch_task_Gibbs.reshape(-1, 1)).reshape(batch_size, n_Gibbs, self.n_action_dim)
        # batch_action_prime_onehot = torch.eye(self.n_action)[batch_action_prime.long()].reshape(batch_size, n_Gibbs, self.action_dim).to(self.device)
        # batch_state_action_Gibbs = torch.concat([batch_state_Gibbs, batch_action_prime_onehot], dim=-1)
        # batch_phi_Gibbs = self.phi(batch_state_action_Gibbs).detach()
        # batch_f_phi_Gibbs = self.critic(batch_phi_Gibbs)
        # batch_task_onehot_Gibbs = torch.eye(self.n_task)[batch_task_Gibbs.long().squeeze(-1)].to(self.device)
        # batch_u_Gibbs = self.u(batch_task_onehot_Gibbs)
        # assert batch_phi_Gibbs.shape == batch_u_Gibbs.shape
        # q_E = torch.sum(batch_f_phi_Gibbs * batch_u_Gibbs, dim=-1, keepdim=False)
        # q_all[:, batch_size:] = q_E
        # assert q_all.shape == (batch_size, batch_size+n_Gibbs)
        # label = torch.eye(batch_size, batch_size+n_Gibbs).to(self.device)
        label = torch.eye(batch_size).to(self.device)
        assert q_all.shape == (batch_size, batch_size) == label.shape
        loss_ctrl = torch.nn.CrossEntropyLoss()(q_all, label)
        # calculate w
        # z_w = self.w(expert_task_onehot)
        # z_mu = self.mu(expert_state)
        # V = self.get_targetV(expert_state, expert_task_onehot)
        # assert V.shape == (batch_size, 1)
        # u_target = z_w + z_mu * V * self.discount * (1 - expert_done)
        # print('u_target', u_target.shape, 'z_w', z_w.shape, 'z_mu', z_mu.shape, 'V', V.shape, 'u1', u1.shape, 'u2', u2.shape)
        # assert u1.shape == u_target.shape
        # assert u2.shape == u_target.shape
        # loss_u = torch.nn.MSELoss()(z_u, u_target)
        # loss_u2 = torch.nn.MSELoss()(u2, u_target)
        # loss_u = (loss_u1 + loss_u2) / 2

        loss_critic_discrete = loss_ctrl
        self.critic_optimizer.zero_grad()
        loss_critic_discrete.backward()
        self.critic_optimizer.step()

        return {
            # 'loss_q': loss_q.item(),
            # 'loss_u': loss_u.item(),
            'loss_critic_discrete': loss_critic_discrete.item(),
            'loss_ctrl_a': loss_ctrl.item(),
        }

    def critic_step_CD(self, batch, s_random, a_random, s_prime_random, task_random):
        expert_state, expert_action, expert_next_state, expert_reward, expert_done, expert_task, expert_next_task = unpack_batch(batch)
        # print('expert_task', expert_task)
        assert expert_state.shape[-1] == self.state_dim
        assert expert_action.shape[-1] == self.action_dim//self.n_action
        assert expert_next_state.shape[-1] == self.state_dim
        assert expert_done.shape[-1] == 1
        expert_task_onehot = torch.eye(self.n_task)[expert_task.long().squeeze(-1)].to(self.device)
        # print('task_onehot', task_onehot.shape)
        assert expert_task_onehot.shape == (expert_state.shape[0], self.n_task)
        expert_action_onehot = torch.eye(self.n_action)[expert_action.long()].reshape(-1, self.action_dim).to(self.device)
        action_prime = self.Gibbs_step(batch)
        z_u = self.u(expert_task_onehot)
        z_phi = self.phi(torch.concat([expert_state, expert_action_onehot], -1)).detach()
        f_phi = self.critic(z_phi)
        q_data = torch.sum(f_phi * z_u, dim=-1, keepdim=True)
        action_prime_onehot = torch.eye(self.n_action)[action_prime.long()].reshape(-1, self.action_dim).to(self.device)
        z_phi_prime = self.phi(torch.concat([expert_state, action_prime_onehot], -1)).detach()
        f_phi_prime = self.critic(z_phi_prime)
        q_E = torch.sum(f_phi_prime * z_u, dim=-1, keepdim=True)
        loss = (q_data - q_E).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return {
            'loss_CD': loss.item(),
            'q_data': q_data.mean().item(),
            'q_E': q_E.mean().item(),
        }
    def Gibbs_step(self, state, action, task, temperature=1):
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.n_action_dim
        assert task.shape[-1] == 1
        assert len(state.shape) == len(action.shape) == len(task.shape) == 2 
        task_onehot = torch.eye(self.n_task)[task.long().squeeze(-1)].to(self.device)
        batch_size = state.shape[0]
        batch_state = state.reshape(batch_size, 1, self.state_dim).repeat(1, self.n_action, 1)
        batch_task_onehot = task_onehot.reshape(batch_size, 1, self.n_task).repeat(1, self.n_action, 1)
        def potential(action):
            # print('action', action.shape)
            action_onehot = torch.eye(self.n_action)[action.long()].reshape(*action.shape[:-1], self.action_dim).to(self.device)
            # print('action_onehot', action_onehot.shape)
            z_phi = self.phi(torch.concat([batch_state, action_onehot], -1))
            f_phi = self.critic(z_phi)
            z_u = self.u(batch_task_onehot)
            q_data = torch.sum(f_phi * z_u, dim=-1, keepdim=False)
            return -q_data
        def sample_one_dimension(action, dim):
            new_action = action.reshape(batch_size, 1, self.n_action_dim).repeat(1, self.n_action, 1)
            new_action[:, :, dim] = torch.arange(self.n_action).reshape(1, -1).repeat(batch_size, 1) 
            new_potential = potential(new_action)
            assert new_potential.shape == (batch_size, self.n_action)
            dist = torch.distributions.Categorical(logits=-new_potential/temperature)
            action_prime_dim = dist.sample()
            # print('action_prime_dim', action_prime_dim.shape)
            assert action_prime_dim.shape == (batch_size, )
            action_prime = action.clone()
            action_prime[:, dim] = action_prime_dim
            return action_prime
        action_prime = action.clone()
        for i in range(self.n_action_dim):
            # print('action_prime:', action_prime)
            action_prime = sample_one_dimension(action_prime, i)
        return action_prime
    def Gibbs_sampling(self, state, action, task, temperature=1, n=100):
        # print('state:', state)
        for i in range(n):
            # print('action:', action)
            action = self.Gibbs_step(state, action, task, temperature)
        return action

    def critic_step(self, batch, s_random, a_random, s_prime_random, task_random):
        """
        Critic update step
        """
        expert_state, expert_action, expert_next_state, expert_reward, expert_done, expert_task, expert_next_task = unpack_batch(batch)
        # print('expert_task', expert_task)
        assert expert_state.shape[-1] == self.state_dim
        assert expert_action.shape[-1] == self.action_dim//self.n_action
        assert expert_next_state.shape[-1] == self.state_dim
        assert expert_done.shape[-1] == 1
        expert_task_onehot = torch.eye(self.n_task)[expert_task.long().squeeze(-1)].to(self.device)
        expert_next_task_onehot = torch.eye(self.n_task)[expert_next_task.long().squeeze(-1)].to(self.device)
        # print('task_onehot', task_onehot.shape)
        assert expert_task_onehot.shape == (expert_state.shape[0], self.n_task)
        batch_size = expert_state.shape[0]
        actor_dist = self.actor(torch.cat([expert_state, expert_task_onehot], -1))
        q_actor_target = torch.gather(actor_dist.logits, dim=-1, index=expert_action.long().reshape(batch_size, -1, 1)).squeeze(-1).sum(-1, keepdim=True)
        assert q_actor_target.shape == (expert_state.shape[0], 1)
        # expert_next_task_onehot = torch.eye(self.n_task)[expert_next_task.long().reshape(-1)].to(self.device)
        # expert_task_onehot_random = torch.eye(self.n_task)[task_random.long().squeeze(-1)].to(self.device)
        # assert expert_task_onehot_random.shape == (expert_state.shape[0], self.n_task)
        # construct negative samples
        # batch_size = expert_state.shape[0]
        # batch_action = torch.arange(self.n_action).reshape(1, 1, self.n_action).repeat(batch_size, self.action_dim//self.n_action, 1)
        # batch_action_onehot = torch.eye(self.n_action)[batch_action.long()].reshape(batch_size, -1, self.n_action).to(self.device)
        # batch_state = expert_state.reshape(batch_size, 1, self.state_dim).repeat(1, self.action_dim, 1)
        # batch_state_action = torch.concat([batch_state, batch_action], dim=-1)
        # batch_state = expert_state.reshape(1, batch_size, self.state_dim)
        # batch_action = expert_action.reshape(batch_size, 1, self.action_dim)
        # batch_action = actor_action.reshape(batch_size, 1, self.action_dim)
        # batch_state_action = torch.concat([batch_state.expand(batch_size, -1, -1), batch_action.expand(-1, batch_size, -1)], dim=-1)
        # batch_z_phi = self.phi(batch_state_action).detach()
        # calculate r
        expert_action_onehot = torch.eye(self.n_action)[expert_action.long()].reshape(-1, self.action_dim).to(self.device)
        z_phi = self.phi(torch.concat([expert_state, expert_action_onehot], -1)).detach() # only need gradient in this place
        # z_phi_random_a = self.phi(torch.concat([expert_state, a_random], -1)).detach()
        z_u = self.u(expert_task_onehot)
        u_l1 = torch.abs(z_u).mean()
        # self.beta.data.clamp_(1,2000)
        # q_all = batch_z_phi @ z_u.T * self.beta
        q_all = torch.sum(z_phi * z_u, dim=-1, keepdim=True)
        assert q_all.shape == (batch_size, 1)
        loss_q = torch.nn.MSELoss()(q_actor_target, q_all)
        # assert q_all.shape == (batch_size, batch_size)
        # label = torch.eye(batch_size).to(self.device)
        # loss_ctrl = torch.nn.CrossEntropyLoss()(q_all, label)
        z_w = self.w(expert_task_onehot)
        rec_r = torch.sum(z_phi * z_w, dim=-1, keepdim=True)
        next_V = self.get_targetV(expert_next_state, expert_next_task_onehot).detach()
        q_bellman_target = rec_r + (1 - expert_done) * self.discount * next_V
        assert next_V.shape == rec_r.shape == q_all.shape == q_bellman_target.shape == (batch_size, 1)
        loss_q_bellman = torch.nn.MSELoss()(q_bellman_target, q_all)
        loss = loss_q + loss_q_bellman
        # print('q_loss', q_loss)
        self.u_optimizer.zero_grad()
        # self.beta_optimizer.zero_grad()
        loss.backward()
        self.u_optimizer.step()
        # for name, param in self.u.named_parameters():
        #     print(name, param.grad)
        # self.beta_optimizer.step()

        return {
            'u_l1': u_l1.item(),
            'loss_q': loss_q.item(),
            'loss_q_bellman': loss_q_bellman.item(),
        }
    def critic_step_QR(self, batch, s_random, a_random, s_prime_random, task_random):
        expert_state, expert_action, expert_next_state, expert_reward, expert_done, expert_task, expert_next_task = unpack_batch(batch)
        # print('expert_task', expert_task)
        assert expert_state.shape[-1] == self.state_dim
        assert expert_action.shape[-1] == self.action_dim//self.n_action
        assert expert_next_state.shape[-1] == self.state_dim
        assert expert_done.shape[-1] == 1
        expert_task_onehot = torch.eye(self.n_task)[expert_task.long().squeeze(-1)].to(self.device)
        expert_next_task_onehot = torch.eye(self.n_task)[expert_next_task.long().squeeze(-1)].to(self.device)
        assert expert_task_onehot.shape == (expert_state.shape[0], self.n_task)
        batch_size = expert_state.shape[0]
        actor_dist = self.actor(torch.cat([expert_state, expert_task_onehot], -1))
        q_actor_target = torch.gather(actor_dist.logits, dim=-1, index=expert_action.long().reshape(batch_size, -1, 1)).squeeze(-1).sum(-1, keepdim=True)
        assert q_actor_target.shape == (expert_state.shape[0], 1)
        expert_action_onehot = torch.eye(self.n_action)[expert_action.long()].reshape(-1, self.action_dim).to(self.device)
        z_phi = self.phi(torch.concat([expert_state, expert_action_onehot], -1)) # only need gradient in this place
        f_phi = self.critic(z_phi)
        z_u = self.u(expert_task_onehot)
        z_w = self.w(expert_task_onehot)
        q_linear = torch.sum(f_phi * z_u, dim=-1, keepdim=True)
        assert q_linear.shape == (batch_size, 1)
        r_linear = torch.sum(f_phi * z_w, dim=-1, keepdim=True)
        assert r_linear.shape == (batch_size, 1)
        # q_linear = qr[:,0:1]
        loss_q = torch.nn.MSELoss()(q_actor_target, q_linear)
        # r_linear = qr[:,1:2]
        assert q_linear.shape == q_actor_target.shape
        next_V = self.get_targetV(expert_next_state, expert_next_task_onehot).detach()
        q_bellman_target = r_linear + (1 - expert_done) * self.discount * next_V
        assert next_V.shape == r_linear.shape == q_linear.shape == q_bellman_target.shape == (batch_size, 1)
        loss_q_bellman = torch.nn.MSELoss()(q_bellman_target, q_linear)
        u_l1 = torch.abs(z_u).mean()
        w_l1 = torch.abs(z_w).mean()
        loss = loss_q + loss_q_bellman + self.lasso_coef * (u_l1 + w_l1)/2 * 100
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return {
            'loss_q': loss_q.item(),
            'loss_q_bellman': loss_q_bellman.item(),
        }

    def update_actor_and_alpha(self, batch):
        """
        Actor update step
        """
        expert_state, expert_action, expert_next_state, expert_reward, expert_done, expert_task, expert_next_task = unpack_batch(batch)
        assert expert_state.shape[-1] == self.state_dim
        # assert expert_action.shape[-1] == self.action_dim//self.n_action    
        assert expert_next_state.shape[-1] == self.state_dim
        assert expert_done.shape[-1] == 1
        expert_task_onehot = torch.eye(self.n_task)[expert_task.long().reshape(-1)].to(self.device)
        # expert_next_task_onehot = torch.eye(self.n_task)[expert_next_task.long().reshape(-1)].to(self.device)
        expert_state_task = torch.cat([expert_state, expert_task_onehot], -1)


        dist = self.actor(expert_state_task)
        expert_log_prob = dist.log_prob(expert_action).sum(-1, keepdim=True)
        negll = -expert_log_prob.mean()

        actor_loss = negll
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        info = {'actor_loss': actor_loss.item(),
                'negll': negll.item()}

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-sample_log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            info['alpha_loss'] = alpha_loss
            info['alpha'] = self.alpha

        return info


    def state_dict(self):
        module_list = {'actor': self.actor.state_dict(),
				'u': self.u.state_dict(),
				'log_alpha': self.log_alpha,
				'phi': self.phi.state_dict(),
				'mu': self.mu.state_dict(),
                'w': self.w.state_dict(),
                'critic': self.critic.state_dict(),}
        print('state dict keys:', module_list.keys())
        return module_list
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.u.load_state_dict(state_dict['u'])
        self.log_alpha = state_dict['log_alpha']
        self.phi.load_state_dict(state_dict['phi'])
        self.mu.load_state_dict(state_dict['mu'])
        self.w.load_state_dict(state_dict['w'])
        print('load state dict keys: actor, critic, u, log_alpha, phi, mu, w')
        # torch.set_printoptions(threshold=torch.inf)
        # print(list(self.phi.parameters()))
        # self.theta.load_state_dict(state_dict['theta'])

    def load_phi_mu(self, state_dict):
        self.phi.load_state_dict(state_dict['phi'])
        self.mu.load_state_dict(state_dict['mu'])
        print('load phi and mu')

    def load_actor(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        print('load actor')

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        # Feature step
        # for _ in range(self.extra_feature_steps + 1):
        batch_1 = buffer.sample(batch_size)
        batch_2 = buffer.sample(batch_size)
        s_random, a_random, s_prime_random, _, _, task_random, next_task_random = unpack_batch(batch_2)
        # s_prime_random = self.obs_dict.sample((batch_size, )).to(self.device)
        feature_info = self.feature_step(batch_1, s_random, a_random, s_prime_random)

        critic_info = self.critic_step_discrete(batch_1, s_random, a_random, s_prime_random, task_random)

        # Actor and alpha step, make the actor closer to softmaxQ
        # actor_info = self.update_actor_and_alpha(batch_1)

        # Update the frozen target models
        self.update_target()

        return {
            **feature_info,
            **critic_info,
            # **actor_info,
        }
    
    def state_likelihood(self, state, action, next_state, kde=False):
        # output the device
        self.phi.eval()
        self.mu.eval()
        with torch.no_grad():
            assert state.shape == action.shape == next_state.shape
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            action_onehot = torch.eye(self.n_action)[action.long()].reshape(-1, self.action_dim).to(self.device)
            z_phi = self.phi(torch.concat([state, action_onehot], -1))
            z_mu_next = self.mu(next_state)
            loglikelihood = (torch.sum(z_phi*z_mu_next, dim=-1))
        return loglikelihood, z_phi, z_mu_next

    def action_loglikelihood(self, state, action, task):
        assert action.shape[-1] == self.action_dim // self.n_action
        # self.phi.eval()
        # self.critic.eval()
        # with torch.no_grad():
            # print('state:', state.shape, 'action:', action.shape, 'task:', task.shape)
            # print('task:', task)
        task_onehot = torch.eye(self.n_task)[task.long()].to(self.device).squeeze(-2)
        # print('task_onehot:', task_onehot.shape)
            # state_task = torch.cat([state, task_onehot], -1)

            # dist = self.actor(state_task)
            # print('dist_mean:', dist.loc[0])
            # print(dist.scale)
            # actor_log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        action_onehot = torch.eye(self.n_action)[action.long()]
        action_onehot = action_onehot.reshape(*action_onehot.shape[:-2], self.action_dim).to(self.device)    
        z_phi = self.phi(torch.concat([state, action_onehot], -1))
            # q = self.critic(torch.cat([z_phi, task_onehot], -1)).squeeze(-1)
        # q = torch.sum(z_phi * self.u(task_onehot)*self.beta, dim=-1, keepdim=False)
        # q = torch.sum(z_phi * self.u(task_onehot), dim=-1, keepdim=False)
        f_phi = self.critic(z_phi)
        z_u = self.u(task_onehot)
        z_w = self.w(task_onehot)
        q = torch.sum(f_phi * z_u, dim=-1, keepdim=False)


            # u1, u2 = self.critic(task_onehot)
            # q1 = torch.sum(z_phi * u1, dim=-1, keepdim=True)
            # q2 = torch.sum(z_phi * u2, dim=-1, keepdim=True)
            # q = torch.min(q1, q2)

            # s_a_feature = self.rescalse_state(self.rescale_state_back(state) + self.rescale_action_back(action))
            # q = self.critic(torch.cat([s_a_feature, task_onehot], -1))
        state_task = torch.cat([state, task_onehot], -1)
        dist = self.actor(state_task)
        actor_log_prob = dist.log_prob(action).sum(-1, keepdim=False)
        assert actor_log_prob.shape == q.shape

        return actor_log_prob, q
    
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
    
    def generate_next_state_discrete_action(self, state, action):
        # print('state:', state.shape, 'action:', action)
        assert action.shape[-1] == self.action_dim // self.n_action
        action_continous = (action - 2)/100
        next_state = state + action_continous
        return next_state

    def step(self, state, action, syllable, temperature=1):
        with torch.no_grad():
            # next_state = self.generate_next_state(state, action)
            # next_state = state + action
            assert state.shape == (1, self.state_dim)
            assert action.shape == (1, self.action_dim//self.n_action)
            next_state = self.generate_next_state_discrete_action(state, action)
            task = torch.FloatTensor([syllable]).to(self.device).reshape(1,1)
            task_onehot = torch.eye(self.n_task)[task.long().squeeze(-1)].to(self.device).reshape(1,-1)
            next_action = self.Gibbs_sampling(state, action, task, temperature=temperature)
            action_onehot = torch.eye(self.n_action)[next_action.long()].reshape(-1, self.action_dim).to(self.device)
            z_phi = self.phi(torch.concat([state, action_onehot], -1))
            mu_next = self.mu(next_state)
            sp_likelihood = torch.sum(z_phi * mu_next, dim=-1)
            f_phi = self.critic(z_phi)
            q = torch.sum(f_phi * self.u(task_onehot), dim=-1, keepdim=False)
        return next_state, next_action, sp_likelihood, q
            


class QR_IRLAgent():
    def __init__(
            self,
            state_dim,
            action_dim,
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
            learnable_temperature=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.n_task = n_task
        self.device = device
        self.critic = DoubleMLP(input_dim=self.state_dim + self.action_dim, # try single task
                          output_dim=1,
                          hidden_dim=hidden_dim,
                            hidden_depth=1).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
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
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=1e-3)
        self.log_alpha = torch.tensor(np.log(1.0)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)
        self.normalize_dict = torch.load(f'./kms/normalize_dict_214.pth')
        self.action_mean = torch.FloatTensor(self.normalize_dict['action_mean']).to(device)
        self.action_std = torch.FloatTensor(self.normalize_dict['action_std']).to(device)
        self.state_mean = torch.FloatTensor(self.normalize_dict['state_mean']).to(device)
        self.state_std = torch.FloatTensor(self.normalize_dict['state_std']).to(device)
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
        action = dist.rsample()
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
        task_onehot = self.task_all[task.long().reshape(-1)].to(self.device)
        # dist = self.actor(torch.cat([state, task_onehot], -1))
        dist = self.actor(state)
        # sample_action = dist.rsample()
        # sample_q1, sample_q2 = self.getQ(state, sample_action)
        # sample_q = torch.min(sample_q1, sample_q2)
        # sample_action_logprob = dist.log_prob(sample_action).sum(-1, keepdim=True)
        # SAC_loss = ((self.alpha) * sample_action_logprob - sample_q).mean()
        # actor_loss = SAC_loss
        ###Behavior Cloning
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_loss = -log_prob.mean()
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
        # critic_info = self.critic_step(buffer.sample(batch_size))
        actor_info = self.update_actor_and_alpha(buffer.sample(batch_size))
        self.update_target()
        return {
            # **critic_info,
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

    def action_loglikelihood(self, state, action, task):
        self.actor.eval()
        task_onehot = self.task_all[task.long().squeeze(1)].to(self.device)
        # dist = self.actor(torch.cat([state, task_onehot], -1))
        dist = self.actor(state)
        print(dist.scale)
        actor_log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return actor_log_prob.mean()

    def generate_next_state(self, state, action):
        original_state = state * self.state_std + self.state_mean
        original_action = action * self.action_std + self.action_mean
        original_next_state = original_state + original_action
        next_state = (original_next_state - self.state_mean) / self.state_std
        return next_state
        
    def step(self, state, task, action):
        print('syllable:', task)
        # state_max, state_min = self.normalize_dict['state_max'], self.normalize_dict['state_min']
        # action_max, action_min = self.normalize_dict['action_max'], self.normalize_dict['action_min']
        with torch.no_grad():
            self.actor.eval()
            self.critic.eval()
            print(state.shape, action.shape)
            next_state = self.generate_next_state(state, action)
            print('next_state:', next_state.shape)
            print(task)
            task_onehot = self.task_all[task].to(self.device).unsqueeze(0)
            # dist = self.actor(torch.cat([next_state, task_onehot], -1))
            dist = self.actor(next_state)
            next_action = dist.sample()
            # q1, q2 = self.getQ(next_state, next_action)
            # q = torch.min(q1, q2)
            # unnormlized_action_logprob = q
        return next_state, next_action, 0, 0

            
class ValueDICEAgent():
    def __init__(
            self,
            state_dim,
            action_dim,
            critic_and_actor_hidden_dim=-1,
            discount=0.99,
            target_update_period=2,
            alpha=0.1,
            device='cuda:0'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.nu_net = MLP(input_dim=state_dim+action_dim,
                          output_dim=1,
                          hidden_dim=critic_and_actor_hidden_dim,
                          hidden_depth=1).to(device)
        self.actor = DiagGaussianActor(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_and_actor_hidden_dim,
            hidden_depth=2,
            log_std_bounds=[-5., 2.],
        ).to(device)
        self.nu_optimizer = torch.optim.Adam(self.nu_net.parameters(), lr=1e-3)
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=1e-3)
        self.log_alpha = torch.tensor(np.log(1.0)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)
        self.normalize_dict = torch.load(f'./kms/normalize_dict_212.pth')
        self.action_mean = torch.FloatTensor(self.normalize_dict['action_mean']).to(device)
        self.action_std = torch.FloatTensor(self.normalize_dict['action_std']).to(device)
        self.steps = 0
        self.discount = discount
        self.replay_regularization = 0.1
    @property
    def alpha(self):
        return self.log_alpha.exp()
    def orthogonal_regularization(model, reg_coef=1e-4):
        """Orthogonal regularization v2.

        See equation (3) in https://arxiv.org/abs/1809.11096.

        Args:
            model: A PyTorch model to apply regularization for.
            reg_coef: Orthogonal regularization coefficient.

        Returns:
            A regularization loss term.
        """
        reg = 0
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                weight = layer.weight
                prod = torch.matmul(weight.t(), weight)
                reg += torch.sum((prod * (1 - torch.eye(prod.shape[0], device=prod.device))) ** 2)
        return reg * reg_coef
    def loss_fn(self, expert_batch, replay_batch):
        torch.autograd.set_detect_anomaly(True)
        expert_state, expert_action, expert_next_state, reward, done, task, next_task = unpack_batch(expert_batch)
        replay_state, replay_action, replay_next_state, reward, done, task, next_task = unpack_batch(replay_batch)

        initial_state = expert_state
        expert_input = torch.cat([expert_state, expert_action], -1)
        rb_input = torch.cat([replay_state, replay_action], -1)

        initial_dist = self.actor(initial_state)
        initial_action = initial_dist.rsample()
        expert_initial_input = torch.cat([initial_state, initial_action], -1)
        expert_nu_0 = self.nu_net(expert_initial_input)
        expert_linear_loss = (1-self.discount) * expert_nu_0.mean()

        expert_next_dist = self.actor(expert_next_state)
        rb_next_dist = self.actor(replay_next_state)
        expert_next_action = expert_next_dist.rsample()
        rb_next_action = rb_next_dist.rsample()
        expert_next_input = torch.cat([expert_next_state, expert_next_action], -1)
        rb_next_input = torch.cat([replay_next_state, rb_next_action], -1)
        expert_next_nu = self.nu_net(expert_next_input)
        rb_next_nu = self.nu_net(rb_next_input)


        expert_nu = self.nu_net(expert_input)
        rb_nu = self.nu_net(rb_input)
        expert_diff = expert_nu - expert_next_nu * self.discount
        rb_diff = rb_nu - rb_next_nu * self.discount
        rb_linear_loss = rb_diff.mean()

        linear_loss = expert_linear_loss * (1-self.replay_regularization) + \
                            rb_linear_loss * self.replay_regularization
        
        rb_expert_diff = torch.cat([rb_diff, expert_diff], 0)
        rb_expert_weights = torch.cat([torch.zeros_like(rb_diff), torch.ones_like(expert_diff)], 0)
        non_linear_loss = torch.sum(torch.softmax(rb_expert_diff, dim=0).detach() * rb_expert_diff * rb_expert_weights, dim=0)

        # Assuming nu_inter is a tensor and self.nu_net is a neural network model
        # nu_inter = torch.cat([expert_input, expert_next_input], 0)
        # nu_output = self.nu_net(nu_inter)

        # Compute gradients
        # nu_grad = torch.autograd.grad(outputs=nu_output, inputs=nu_inter, 
        #                             grad_outputs=torch.ones_like(nu_output))[0]

        # Compute gradient penalty
        # nu_grad_penalty = torch.mean((torch.norm(nu_grad, dim=-1, keepdim=True) - 1) ** 2)
        # actor_regularization = self.orthogonal_regularization(self.actor)

        loss = non_linear_loss - expert_linear_loss
        return loss, non_linear_loss, expert_linear_loss
    

    def train(self, expert_buffer, replay_buffer, batch_size):
        expert_batch = expert_buffer.sample(batch_size)
        replay_batch = replay_buffer.sample(batch_size)
        nu_loss, non_linear_loss, linear_loss = self.loss_fn(expert_batch, replay_batch)
        self.nu_optimizer.zero_grad()
        nu_loss.backward()
        self.nu_optimizer.step()
        nu_loss, non_linear_loss, linear_loss = self.loss_fn(expert_batch, replay_batch)
        pi_loss = -nu_loss
        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_optimizer.step()
        return {
            'nu_loss': nu_loss.item(),
            'pi_loss': pi_loss.item(),
            'non_linear_loss': non_linear_loss.item(),
            'linear_loss': linear_loss.item()
        }
    def generate_next_state(self, state, action):
        original_action = action * self.action_std + self.action_mean
        return state + original_action

class SimpleWorldModel():
    """
    SAC with VAE learned latent features
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space=None,
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
        self.steps = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.device = device
        self.n_task = n_task
        self.normalize_dict = torch.load(f'./kms/normalize_dict_allnormalized_augment.pth')
        self.action_mean = torch.FloatTensor(self.normalize_dict['action_mean']).to(device)
        self.action_std = torch.FloatTensor(self.normalize_dict['action_std']).to(device)
        self.state_mean = torch.FloatTensor(self.normalize_dict['state_mean']).to(device)
        self.state_std = torch.FloatTensor(self.normalize_dict['state_std']).to(device)
        self.state_min = torch.FloatTensor(self.normalize_dict['state_min']).to(device)
        self.state_max = torch.FloatTensor(self.normalize_dict['state_max']).to(device)
        self.action_min = torch.FloatTensor(self.normalize_dict['action_min']).to(device)
        self.action_max = torch.FloatTensor(self.normalize_dict['action_max']).to(device)

        self.p_sprime = pyd.Normal(loc=torch.zeros(state_dim).to(device), scale=torch.ones(state_dim).to(device))
        # self.likelihood_network = DiagGaussianActor(
        #     obs_dim=state_dim + action_dim,
        #     action_dim=state_dim,
        #     hidden_dim=hidden_dim,
        #     hidden_depth=1,
        #     log_std_bounds=[-10., 2.],
        # ).to(device)
        self.likelihood_network = MLP(input_dim=state_dim + action_dim + state_dim,
                                                           output_dim=1,
                                                           hidden_dim=hidden_dim,
                                                           hidden_depth=2).to(device)
        self.likelihood_optimizer = torch.optim.Adam(list(self.likelihood_network.parameters()),
                                                    weight_decay=0, lr=1e-4, betas=[0.9, 0.999])

    # def likelihood_step(self, batch):
    #     state, action, next_state, reward, _, task, next_task = unpack_batch(batch)
    #     log_prob_sprime = self.log_likelihood(state, action, next_state)
    #     # print(log_prob_sprime)
    #     negll = -log_prob_sprime.mean()
    #     self.likelihood_optimizer.zero_grad()
    #     negll.backward()
    #     self.likelihood_optimizer.step()
    #     return {
    #         'negll': negll.item()
    #     }
    # def log_likelihood(self, state, action, next_state):
    #     assert state.shape[-1] == self.state_dim
    #     assert action.shape[-1] == self.action_dim
    #     assert next_state.shape[-1] == self.state_dim
    #     next_s_dist = self.likelihood_network(torch.concat([state, action], -1))
    #     log_prob_sprime = next_s_dist.log_prob(next_state).sum(-1, keepdim=True)
    #     return log_prob_sprime
    def contrastive_likelihood_step(self, batch_1, batch_2):
        state, action, next_state, reward, _, task, next_task = unpack_batch(batch_1)
        s_random, a_random, s_prime_random, _, _, task_random, next_task_random = unpack_batch(batch_2)
        assert state.shape[-1] == self.state_dim == s_random.shape[-1] == next_state.shape[-1] == s_prime_random.shape[-1]
        assert action.shape[-1] == self.action_dim == a_random.shape[-1]
        positive_log_prob_sprime = self.log_likelihood_part(state, action, next_state)
        pos_loss = torch.nn.BCEWithLogitsLoss()(positive_log_prob_sprime, torch.ones_like(positive_log_prob_sprime))
        random_log_prob_sprime = self.log_likelihood_part(state, a_random, next_state)
        neg_loss_1 = torch.nn.BCEWithLogitsLoss()(random_log_prob_sprime, torch.zeros_like(random_log_prob_sprime))
        random_log_prob_sprime = self.log_likelihood_part(state, action, s_prime_random)
        neg_loss_2 = torch.nn.BCEWithLogitsLoss()(random_log_prob_sprime, torch.zeros_like(random_log_prob_sprime))
        loss = pos_loss + neg_loss_1 + neg_loss_2
        # loss = F.binary_cross_entropy_with_logits(log_prob_sprime, torch.ones_like(log_prob_sprime))
        self.likelihood_optimizer.zero_grad()
        loss.backward()
        self.likelihood_optimizer.step()
        return {
            'loss': loss.item()
        }
    def log_likelihood_part(self, state, action, next_state):
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim
        return self.likelihood_network(torch.concat([state, action, next_state], -1))
    def feature_step(self, batch_1, batch_2):
        state, action, next_state, reward, _, task, next_task = unpack_batch(batch_1)
        s_random, a_random, _, _, _, task_random, next_task_random = unpack_batch(batch_2)
        s_prime_random = self.p_sprime.sample((s_random.shape[0],)).to(self.device)
        assert state.shape[-1] == self.state_dim == s_random.shape[-1] == next_state.shape[-1] == s_prime_random.shape[-1]
        assert action.shape[-1] == self.action_dim == a_random.shape[-1]
        positive_log_prob_sprime = self.log_likelihood_part(state, action, next_state)
        model_loss_1 = - 2 * positive_log_prob_sprime.mean()

        random_log_prob_sprime = self.log_likelihood_part(s_random, a_random, s_prime_random)
        model_loss_2 = ((random_log_prob_sprime)**2).mean()
        loss = model_loss_1 + model_loss_2
        self.likelihood_optimizer.zero_grad()
        loss.backward()
        self.likelihood_optimizer.step()
        return {
            'model_loss_1': model_loss_1.item(),
            'model_loss_2': model_loss_2.item(),
            'loss': loss.item()
        }
    def log_likelihood_all(self, state, action, next_state):
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim
        logf = torch.log(self.likelihood_network(torch.concat([state, action, next_state], -1)))
        logPrior = self.p_sprime.log_prob(next_state).sum(-1, keepdim=True)
        return logf + logPrior
    def state_dict(self):
        return {'likelihood_network': self.likelihood_network.state_dict()}
    def load_state_dict(self, state_dict):
        self.likelihood_network.load_state_dict(state_dict['likelihood_network'])

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1
        batch_1 = buffer.sample(batch_size)
        batch_2 = buffer.sample(batch_size)
        feature_info = self.contrastive_likelihood_step(batch_1, batch_2)
        return {
            **feature_info
        }

class RandomFeatureModel():

    def __init__(
            self,
            state_dim,
            action_dim,
            discount=0.99,
            device='cuda:0',
            n_task=3, **kwargs):
        self.steps = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.device = device
        self.n_task = n_task
        # self.normalize_dict = torch.load(f'./kms/normalize_dict_allnormalized_augment.pth')
        # self.action_mean = torch.FloatTensor(self.normalize_dict['action_mean']).to(device)
        # self.action_std = torch.FloatTensor(self.normalize_dict['action_std']).to(device)
        # self.state_mean = torch.FloatTensor(self.normalize_dict['state_mean']).to(device)
        # self.state_std = torch.FloatTensor(self.normalize_dict['state_std']).to(device)
        # self.state_min = torch.FloatTensor(self.normalize_dict['state_min']).to(device)
        # self.state_max = torch.FloatTensor(self.normalize_dict['state_max']).to(device)
        # self.action_min = torch.FloatTensor(self.normalize_dict['action_min']).to(device)
        # self.action_max = torch.FloatTensor(self.normalize_dict['action_max']).to(device)
        self.sample_times = 8
        self.p_omega = pyd.Normal(loc=0, scale=1)
        self.p_b = pyd.Uniform(low=0, high=2*np.pi)
        # self.omega_list = torch.load('./kms/omega_list.pt').to(self.device)
        # self.b_list = torch.load('./kms/b_list.pt').to(self.device)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        self.omega_list = torch.randn(state_dim, self.sample_times*state_dim).to(self.device)
        self.b_list = torch.randn(1, self.sample_times*state_dim).to(self.device)
        print('Omega:', self.omega_list.mean(), self.omega_list.std())
        print('b:', self.b_list.mean(), self.b_list.std())
        self.phi = RFFMLP_notrain(input_dim=state_dim, output_dim=state_dim*self.sample_times).to(device)
        # print('Phi:', self.phi.l1.weight)
        # print('Omega:', self.omega_list)    
        self.mu = copy.deepcopy(self.phi)
    def state_likelihood(self, state, action, next_state):
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim
        batch_size = state.shape[0]
        # original_state = state * self.state_std + self.state_mean
        # original_action = action * self.action_std + self.action_mean
        # original_next_state = next_state
        # psi_sa = ((original_state + original_action)-self.state_mean) / self.state_std
        psi_sa = state + action
        print(psi_sa)
        # nu_sprime = (original_next_state)
        nu_sprime = next_state
        print(nu_sprime)
        # print(((psi_sa-nu_sprime)**2).mean())
        # print(psi_sa, nu_sprime)
        z_phi = self.phi(psi_sa)
        z_mu = self.mu(nu_sprime)
        # z_phi = torch.cos(psi_sa @ self.phi.l1.weight.T + self.phi.l1.bias)
        # z_mu = torch.cos(nu_sprime @ self.mu.l1.weight.T + self.mu.l1.bias)
        # print(z_phi.shape)
        # z_phi = torch.cos(psi_sa @ self.omega_list + self.b_list)
        # z_mu = torch.cos(nu_sprime @ self.omega_list + self.b_list)
        assert z_phi.shape == z_mu.shape == (batch_size, self.state_dim*self.sample_times)
        prob = torch.sum(z_phi * z_mu, dim=-1, keepdim=True)
        # prob = torch.sum(z_phi**2, dim=-1, keepdim=True) 
        assert prob.shape == (batch_size, 1)
        return prob