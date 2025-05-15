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
from utils.util import unpack_batch, MLP
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
from plot import plot_auc, agent_likelihood_fn, linear_loglikelihood, cal_plot_auc, save_fig

class Finetune_Agent():

    def __init__(self, args, dataset, spedersacagent):
        self.args = args
        self.dataset = dataset
        self.agent = spedersacagent
        self.device = args.device
    def fit_soft_syllable_batch(self, times=250):
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
        file_path = f'./kms/fit_soft_syllable_batch.txt'
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
            # mle
            state_all = state.reshape(sample_len, 1, agent.state_dim).repeat(1, n_contrastive_sample+1, 1)
            # map
            # state_1 = state.reshape(sample_len, 1, agent.state_dim)
            # state_2 = state_2.reshape(sample_len, n_contrastive_sample, agent.state_dim)
            # state_all = torch.concat([state_1, state_2], dim=1)
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
            with open(file_path, 'a') as f:
                f.write(f'iter {k}, pos_logll: {positive_logll}, neg_logll: {negative_logll}, pos_logll_lr: {positive_logll_linear}, neg_logll_lr: {negative_logll_linear}\n')
        plot_auc(pos_logll, neg_logll, pos_logll_lr, neg_logll_lr, f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/auc.pdf')
    
    def fit_whole_dataset(self, critic_save_path, u_vec_save_path, suffix, times=1000):
        dataset = self.dataset
        agent = self.agent
        args = self.args
        # replay_buffer, state_dim, action_dim, n_task = load_all_keymoseq('test', args.dir, args.device)
        device = 'cuda:0'
        dataset = buffer.ReplayBuffer(agent.state_dim, agent.action_dim, 1000000, device)
        dataset.load_state_dict(torch.load('./kms/replay_buffer_all_normalized.pth'))
        print('dataset size:', dataset.size, dataset.state.shape)
        np.random.seed(3)
        sample_len = 1000
        n_sample = 16
        n_contrastive_sample = n_sample-1
        label = torch.zeros((sample_len, n_sample)).to(device)
        label[:,0] = 1
        label.requires_grad = False
        phi_dim = 64
        u_vec = torch.randn((dataset.size, phi_dim)).to(device)
        self.u_vec = u_vec
        arr = np.lib.format.open_memmap(
                u_vec_save_path,
                mode='w+',
                dtype='float32',
                shape=u_vec.cpu().numpy().shape
            )
        print(f'start to write into {u_vec_save_path}')
        arr[:] = u_vec.cpu().numpy()
        iteration = 50
        n_step = 1000
        agent.phi = agent.phi.to(device)
        agent.critic = MLP(input_dim=agent.feature_dim,
                                output_dim=phi_dim,
                                hidden_dim=16,
                                hidden_depth=0,
                                bias=True,
                                output_mod=nn.ELU()).to(device)
        critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=1e-3)
        
        # sigma = 1
        # l = 10
        # coef = 1e-5
        # var_noise = 5e-5

        for k in range(times):
            print(f'iter {k}')
            sample_idx = np.random.randint(0, dataset.size-sample_len) + np.arange(sample_len)
            state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
            initial_u = u_vec[sample_idx]
            u_matrix = initial_u.clone().detach().requires_grad_() # [sample_len, feature_dim]


            # t = torch.arange(u_matrix.shape[0]).to(device)
            # diff = t.reshape(1,-1) - t.reshape(-1,1)
            # K = torch.exp(-diff**2/2/l**2) * sigma**2 + var_noise*torch.eye(u_matrix.shape[0]).to(device)
            # K_inv = torch.linalg.inv(K)


            state_2, action_2, next_state_2, reward_2, done_2, task_2, next_task_2 = unpack_batch(dataset.sample(sample_len*n_contrastive_sample))
            action_2 = action_2.reshape(sample_len, n_contrastive_sample, agent.action_dim)
            action_1 = action.reshape(sample_len, 1, agent.action_dim)
            action_all = torch.concat([action_1, action_2], dim=1)
            # mle
            state_all = state.reshape(sample_len, 1, agent.state_dim).repeat(1, n_sample, 1)
            # map
            # state_1 = state.reshape(sample_len, 1, agent.state_dim)
            # state_2 = state_2.reshape(sample_len, n_contrastive_sample, agent.state_dim)
            # state_all = torch.concat([state_1, state_2], dim=1)
            s_a = torch.concat([state_all, action_all], dim=-1)
            z_phi = agent.phi(s_a).detach()
            assert z_phi.shape == (sample_len, n_sample, agent.feature_dim)
            u_optimizer = torch.optim.Adam([u_matrix], lr=1e-3)
            def loss_fn(u_matrix, f_phi_matrix):
                Q = torch.sum(f_phi_matrix * u_matrix.unsqueeze(1), dim=-1)
                assert Q.shape == label.shape
                loss_ctrl = nn.CrossEntropyLoss()(Q, label)
                neglogprior = (torch.diff(u_matrix, dim=0)**2).mean() * 10
                # Gaussian process
                # neglogprior = torch.diag(u_matrix.T @ K_inv @ u_matrix).mean()*coef
                loss_reg = (u_matrix.abs()).mean() * 0.1
                # loss_reg = torch.zeros_like(neglogprior)
                loss = loss_ctrl + neglogprior + loss_reg
                return loss, loss_ctrl, neglogprior, loss_reg
            for i in range(iteration):
                f_phi_matrix = agent.critic(z_phi).detach()
                assert f_phi_matrix.shape == (sample_len, n_sample, phi_dim)
                assert u_matrix.shape == (sample_len, phi_dim)
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
            u_vec[sample_idx] = u_matrix.detach()
            arr[sample_idx] = u_matrix.detach().cpu().numpy()
            torch.save(agent.critic.state_dict(), critic_save_path)
            np.save(u_vec_save_path.replace('.npy', '_backup.npy'), u_vec.detach().cpu().numpy())
            self.u_vec = u_vec
            with open(f'./kms/fit_whole_dataset_{suffix}.txt', 'a') as f:
                f.write(f'iter {k}, loss: {loss.item():.4f}, loss_ctrl: {loss_ctrl.mean().item():.4f}, neglogprior: {neglogprior.item():.4f}, loss_reg: {loss_reg.item():.4f}\n')
        return u_vec
    def sample_plot_auc(self, suffix, sample_len=250, times=50):
        agent = self.agent
        args = self.args
        device = 'cuda:0'
        agent.phi = agent.phi.to(device)
        phi_dim = 512
        agent.critic = MLP(input_dim=agent.feature_dim,
                                output_dim=phi_dim,
                                hidden_dim=16,
                                hidden_depth=0,
                                bias=True,
                                output_mod=nn.ELU()).to(device)
        dataset = buffer.ReplayBuffer(agent.state_dim, agent.action_dim, 1000000, device)
        dataset.load_state_dict(torch.load('./kms/replay_buffer_all_normalized.pth'))
        print('dataset size:', dataset.size, dataset.state.shape)
        self.u_vec = torch.FloatTensor(np.load(f'./kms/u_vec_{suffix}.npy')).to(device)
        print(f'u_vec:./kms/u_vec_{suffix}.npy', type(self.u_vec))
        self.agent.critic.load_state_dict(torch.load(f'./kms/critic_{suffix}.pth'))
        print(f'critic:./kms/critic_{suffix}.pth')
        device = 'cuda:0'
        auc_agents = np.zeros((times, ))
        auc_linears = np.zeros((times, ))
        for i in range(times):
            sample_idx = np.random.randint(0, self.dataset.size-sample_len) + np.arange(sample_len)
            state, action, next_state, reward, done, task, next_task = unpack_batch(self.dataset.take(sample_idx))
            u_matrix = self.u_vec[sample_idx]
            initial_u = torch.zeros_like(u_matrix).to(self.device)
            auc_agent, auc_linear = cal_plot_auc(state, action, task, initial_u, 
                            u_matrix, self.dataset, self.agent, batch_size=sample_len, 
                            save_path=f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/auc_longclip_{sample_idx[0]}+{sample_len}_{suffix}.pdf',
                            seed=i, device=device)
            auc_agents[i] = auc_agent
            auc_linears[i] = auc_linear
        fig, axis = plt.subplots(1, 1, figsize=(5, 5))
        axis.errorbar(np.arange(1), auc_agents.mean(), yerr=auc_agents.std(), label='agent')
        axis.errorbar(np.arange(1)+1, auc_linears.mean(), yerr=auc_linears.std(), label='linear')
        axis.set_title(f'auc: {auc_agents.mean():.4f} +/- {auc_agents.std():.4f}, linear: {auc_linears.mean():.4f} +/- {auc_linears.std():.4f}')
        axis.legend()
        save_fig(f'./kms/auc_longclip_whole_{suffix}.png')
        
    def load_broken_npy(self, path, dtype='float32'):
        import struct
        agent = self.agent
        device = 'cpu'
        dataset = buffer.ReplayBuffer(agent.state_dim, agent.action_dim, 1000000, device)
        dataset.load_state_dict(torch.load('./kms/replay_buffer_all_normalized.pth'))
        print('dataset size:', dataset.size, dataset.state.shape)
        shape = (dataset.size, 512)
        """
        Load a .npy file whose header may be malformed, by
        hand-skipping the header and memmapping the remainder.
        
        Parameters
        ----------
        path : str
        Path to the .npy file.
        dtype : np.dtype or str
        The true dtype (e.g. 'float32').
        shape : tuple of int
        The true shape of the array.
        
        Returns
        -------
        arr : np.ndarray
        The array, reshaped to `shape`.
        """
        with open(path, 'rb') as f:
            # 1) Skip the magic string: 6 bytes
            magic = f.read(6)  
            # 2) Read version number (2 bytes)
            major, minor = f.read(2)
            # 3) Read the header length (2-byte little-endian uint16 for v1.0 files)
            header_len = struct.unpack('<H', f.read(2))[0]
            # 4) Skip the rest of the header
            f.seek(header_len, os.SEEK_CUR)
            offset = f.tell()

        # Now memory‐map the raw data past the header
        data = np.memmap(path, dtype=dtype, mode='r', offset=offset)
        u_vec = np.asarray(data).reshape(shape)
        print(f'u_vec shape: {u_vec.shape}')
        print(u_vec[:20])
        return u_vec