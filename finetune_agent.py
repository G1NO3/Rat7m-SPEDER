# import gym.spaces
import numpy as np
import torch
import gym
import argparse
import os, glob
from PIL import Image
from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.vlsac import vlsac_agent
from agent.ctrlsac import ctrlsac_agent
from agent.diffsrsac import diffsrsac_agent
from agent.spedersac import spedersac_agent
from main import load_rat7m, load_halfcheetah, load_keymoseq, load_all_keymoseq
from utils.util import unpack_batch
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, ttest_rel
from torch import nn, optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# import umap
import seaborn as sns
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from torch.nn import functional as F
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import pickle
from functools import partial
from plot import plot_auc, agent_likelihood_fn, linear_loglikelihood

class Finetune_Agent():

    def __init__(self, args, dataset, spedersacagent):
        self.args = args
        self.dataset = dataset
        self.agent = spedersacagent
        self.device = args.device
    def fit_soft_syllable_batch(self, times=20):
        dataset = self.dataset
        agent = self.agent
        args = self.args
        # replay_buffer, state_dim, action_dim, n_task = load_all_keymoseq('test', args.dir, args.device)
        pos_logll = np.zeros((times,))
        neg_logll = np.zeros((times,))
        pos_logll_lr = np.zeros((times,))
        neg_logll_lr = np.zeros((times,))
        np.random.seed(3)
        sample_len = 100
        n_contrastive_sample = 16-1
        iteration = 50
        n_step = 1000
        label = torch.zeros((sample_len, n_contrastive_sample+1))
        label[:,0] = 1
        label.requires_grad = False
        for k in range(times):
            sample_idx = np.random.randint(0, dataset.size-sample_len) + np.arange(sample_len)
            state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
            task_onehot = F.one_hot(task.reshape(-1).long(), num_classes=agent.n_task).float()
            initial_u = agent.u(task_onehot)
            # state_matrix_all[k] = state.detach().cpu().numpy()
            # action_matrix_all[k] = action.detach().cpu().numpy()
            # task_matrix_all[k] = task.detach().cpu().numpy()
            # u_matrix_all[k] = initial_u.detach().cpu().numpy()
            u_matrix = initial_u.clone().detach().requires_grad_() # [sample_len, feature_dim]
            state_2, action_2, next_state_2, reward_2, done_2, task_2, next_task_2 = unpack_batch(dataset.sample(sample_len*n_contrastive_sample))
            action_2 = action_2.reshape(sample_len, n_contrastive_sample, agent.action_dim)
            action_1 = action.reshape(sample_len, 1, agent.action_dim)
            action_all = torch.concat([action_1, action_2], dim=1)
            state_all = state.reshape(sample_len, 1, agent.state_dim).repeat(1, n_contrastive_sample+1, 1)
            s_a = torch.concat([state_all, action_all], dim=-1)
            z_phi = agent.phi(s_a).detach()
            assert z_phi.shape == (sample_len, n_contrastive_sample+1, agent.feature_dim)
            u_optimizer = torch.optim.Adam([u_matrix], lr=1e-3)
            critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=1e-3)
            def loss_fn(u_matrix, f_phi_matrix):
                Q = torch.sum(f_phi_matrix * u_matrix.unsqueeze(1), dim=-1)
                assert Q.shape == label.shape
                loss_ctrl = nn.CrossEntropyLoss()(Q, label)
                neglogprior = (torch.diff(u_matrix, dim=0)**2).mean() * 100
                loss_reg = (u_matrix.abs()).mean() * 1
                # loss_reg = torch.zeros_like(neglogprior)
                loss = loss_ctrl + neglogprior + loss_reg
                return loss, loss_ctrl, neglogprior, loss_reg
            for i in range(iteration):
                f_phi_matrix = agent.critic(z_phi).detach()
                assert f_phi_matrix.shape == (sample_len, n_contrastive_sample+1, agent.feature_dim)
                assert u_matrix.shape == (sample_len, agent.feature_dim)
                for j in range(n_step):
                    loss, loss_ctrl, neglogprior, loss_reg = loss_fn(u_matrix, f_phi_matrix)
                    u_optimizer.zero_grad()
                    loss.backward()
                    u_optimizer.step()
                for j in range(n_step):
                    f_phi_matrix = agent.critic(z_phi)
                    loss, loss_ctrl, neglogprior, loss_reg = loss_fn(u_matrix, f_phi_matrix)
                    critic_optimizer.zero_grad()
                    loss.backward()
                    critic_optimizer.step()
                print(f'iter {i}, loss: {loss.item():.4f}, loss_ctrl: {loss_ctrl.mean().item():.4f}, neglogprior: {neglogprior.item():.4f}, loss_reg: {loss_reg.item():.4f}')
            f_phi_matrix = agent.critic(z_phi).detach()
            sample_idx = np.random.randint(0, dataset.size-sample_len)+np.arange(sample_len)
            state_2, action_2, next_state_2, reward_2, done_2, task_2, next_task_2 = unpack_batch(dataset.take(sample_idx))
            positive_logll = agent_likelihood_fn(agent, state, action, task, u_matrix).detach().cpu().numpy().mean()
            negative_logll = agent_likelihood_fn(agent, state, action_2, task, u_matrix).detach().cpu().numpy().mean()
            lr = pickle.load(open('./kms/linear_all.pkl', 'rb'))
            positive_logll_linear = linear_loglikelihood(state, action, task, lr).detach().cpu().numpy().mean()
            negative_logll_linear = linear_loglikelihood(state, action_2, task, lr).detach().cpu().numpy().mean()
            pos_logll_lr[k] = positive_logll_linear
            neg_logll_lr[k] = negative_logll_linear
            pos_logll[k] = positive_logll
            neg_logll[k] = negative_logll
        plot_auc(pos_logll, neg_logll, pos_logll_lr, neg_logll_lr, f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/auc.pdf')

