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
from agent.spedersac import spedersac_agent, iql_agent
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
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from sklearn.metrics import roc_auc_score, roc_curve
from plot import save_fig, get_edges, plot_gif_onefig, rasterize_figure, pair_gif_and_u, \
              plot_auc, plot_u, cal_plot_auc, agent_likelihood_fn, linear_loglikelihood
from finetune_agent import Finetune_Agent
from utils.util import MLP
from agent.opal import opal_agent
import math


def fit_train_test(args, dataset, agent):
  times = 10
  train_auc_agent = np.zeros((times, ))
  train_auc_linear = np.zeros((times, ))
  test_auc_agent = np.zeros((times, ))
  test_auc_linear = np.zeros((times, ))
  np.random.seed(4)
  train_idxs = np.random.randint(0, dataset.size-1000, size=times) 
  test_idxs = np.random.randint(0, dataset.size-1000, size=times)
  f_path = f'./kms/auc_average.txt'
  f = open(f_path, 'w')
  for i in range(times):
    train_idx = train_idxs[i]
    print('train_idx:', train_idx)
    auc_agent_mean, auc_agent_std, auc_linear_mean, auc_linear_std = \
      fit_soft_syllable(args, dataset, agent, mode='train', initial_sample_idx=train_idx, train_idx=None)
    train_auc_agent[i] = auc_agent_mean
    train_auc_linear[i] = auc_linear_mean
    print('train_auc_agent:', train_auc_agent[i], 'train_auc_linear:', train_auc_linear[i])
    test_idx = test_idxs[i]
    print('test_idx:', test_idx)
    auc_agent_mean, auc_agent_std, auc_linear_mean, auc_linear_std = \
      fit_soft_syllable(args, dataset, agent, mode='test', initial_sample_idx=test_idx, train_idx=train_idx)
    test_auc_agent[i] = auc_agent_mean
    test_auc_linear[i] = auc_linear_mean
    print('test_auc_agent:', test_auc_agent[i], 'test_auc_linear:', test_auc_linear[i])
    f.write(f'{i}\n train_idx:{train_idx} {train_auc_agent[i]} {train_auc_linear[i]}\n test_idx:{test_idx} {test_auc_agent[i]} {test_auc_linear[i]}\n')
    f.flush()
  f.close()

  fig, axis = plt.subplots(1, 2, figsize=(10, 5))
  np.save(f'./kms/train_auc_agent.npy', train_auc_agent)
  np.save(f'./kms/train_auc_linear.npy', train_auc_linear)
  axis[0].errorbar(np.arange(1), train_auc_agent.mean(), yerr=train_auc_agent.std(), label='agent')
  axis[0].errorbar(np.arange(1)+1, train_auc_linear.mean(), yerr=train_auc_linear.std(), label='linear')
  axis[0].set_title(f'train auc: {train_auc_agent.mean():.4f} +/- {train_auc_agent.std():.4f}, linear: {train_auc_linear.mean():.4f} +/- {train_auc_linear.std():.4f}')
  axis[0].legend()
  axis[1].errorbar(np.arange(1), test_auc_agent.mean(), yerr=test_auc_agent.std(), label='agent')
  axis[1].errorbar(np.arange(1)+1, test_auc_linear.mean(), yerr=test_auc_linear.std(), label='linear')
  axis[1].set_title(f'test auc: {test_auc_agent.mean():.4f} +/- {test_auc_agent.std():.4f}, linear: {test_auc_linear.mean():.4f} +/- {test_auc_linear.std():.4f}')
  axis[1].legend()
  save_fig(f'./kms/auc_average.png')

  

def fit_soft_syllable(args, dataset, agent, mode, initial_sample_idx, train_idx):
  # replay_buffer, state_dim, action_dim, n_task = load_all_keymoseq('test', args.dir, args.device)
  device = 'cuda:0'

  sample_len = 250
  n_sample = 16
  z_phi_matrix = torch.zeros((n_sample, sample_len, agent.feature_dim))
  # print('dataset device:', dataset.device)
  for i in range(n_sample):
    sample_idx = np.random.randint(0, dataset.size-sample_len) + np.arange(sample_len)
    state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
    if i == 0:
      sample_idx = initial_sample_idx + np.arange(sample_len)
      # sample_idx = 68290 + np.arange(sample_len)
      # sample_idx = 38772 + np.arange(sample_len)
      # sample_idx = 4610 + np.arange(sample_len)
      # sample_idx = 99596 + np.arange(sample_len)
      state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
      # print('sample_idx:', sample_idx)
      # print('task:', task.reshape(-1))
      task_onehot = F.one_hot(task.reshape(-1).long(), num_classes=agent.n_task).float()
      initial_u = agent.u(task_onehot)
      u_matrix = initial_u.clone().detach().requires_grad_() # [sample_len, feature_dim]
      initial_state = state.clone().detach()
      initial_task = task.clone().detach()
      initial_action = action.clone().detach()
      initial_sample_idx = sample_idx[0]
    # print('action:', action.device, 'initial_state:', initial_state.device)
    # devices = {p.device for p in agent.phi.parameters()}
    # print('devices:', devices)
    z_phi = agent.phi(torch.concat([initial_state, action], -1))
    z_phi_matrix[i] = z_phi
  z_phi_matrix = z_phi_matrix.to(device)
  # u_optimizer = torch.optim.Adam([u_matrix]+list(agent.critic.parameters()), lr=1e-4)
  # print('init_state:', initial_state.shape)
  phi_dim = 64
  u_matrix = torch.randn((u_matrix.shape[0], phi_dim)).to(device).requires_grad_()

  initial_u = u_matrix.clone().detach()
  # print('u_matrix:', u_matrix.shape)  
  
  agent.critic = MLP(input_dim=agent.feature_dim,
                                output_dim=phi_dim,
                                hidden_dim=16,
                                hidden_depth=0,
                                bias=True,
                                output_mod=nn.ELU()).to(device)
  if mode == 'train':
    critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=1e-3)
  elif mode == 'test':
    u_matrix = torch.FloatTensor(np.load(f'./kms/u_matrix_{train_idx}_64to64_mle_sparse01_grw10_train1.npy')).to(device).requires_grad_()
    agent.critic.load_state_dict(torch.load(f'./kms/critic_{train_idx}_64to64_mle_sparse01_grw10_train1.pth'))
    

  u_optimizer = torch.optim.Adam([u_matrix], lr=1e-3)
  # step = 50000
  iteration = 50
  n_step = 1000
  label = torch.zeros((sample_len, n_sample)).to(device)
  label[:,0] = 1
  label.requires_grad = False
  z_phi_matrix = z_phi_matrix.detach()
  # f_phi_matrix = f_phi_matrix.detach()
  initial_f_phi_matrix = agent.critic(z_phi_matrix).detach()
  u_matrix_list = torch.zeros((iteration, sample_len, phi_dim))


  # sigma = 1
  # l = 10
  # coef = 1e-5
  # var_noise = 5e-5
  # t = torch.arange(u_matrix.shape[0]).to(device)
  # diff = t.reshape(1,-1) - t.reshape(-1,1)
  # K = torch.exp(-diff**2/2/l**2) * sigma**2 + var_noise*torch.eye(u_matrix.shape[0]).to(device)
  # K_inv = torch.linalg.inv(K)
  # fig, ax = plt.subplots(1,1, figsize=(5,5))
  # im = ax.imshow(K_inv.detach().cpu().numpy(), cmap='hot', interpolation='nearest')  
  # ax.set_title('kernel matrix')
  # fig.colorbar(im, ax=ax, label='Intensity')
  # save_fig(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/kernel_matrix.png')
  def loss_fn(u_matrix, f_phi_matrix):
    Q = torch.sum(f_phi_matrix * u_matrix.unsqueeze(0), dim=-1).T
    assert Q.shape == label.shape
    loss_ctrl = nn.CrossEntropyLoss()(Q, label)

    neglogprior = (torch.diff(u_matrix, dim=0)**2).mean() * 10
    # neglogprior = torch.diag(u_matrix.T @ K_inv @ u_matrix).mean()*coef


    loss_reg = (u_matrix.abs()).mean() * 0.1
    # loss_reg = torch.zeros_like(neglogprior)
    loss = loss_ctrl + neglogprior + loss_reg
    return loss, loss_ctrl, neglogprior, loss_reg
  for i in range(iteration):
    # if i % 10 == 0:
    #   fig, axis = plt.subplots(1, 1, figsize=(15, 5))
    #   for j in range(u_matrix.shape[1]):
    #     axis.plot(u_matrix[:, j].detach().cpu().numpy(), label=f'{j}')
    #   axis.set_title(f'iter {i}')
    #   save_fig(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/u_matrix_{i}.png')
    f_phi_matrix = agent.critic(z_phi_matrix).detach()
    for j in range(n_step):
      loss, loss_ctrl, neglogprior, loss_reg = loss_fn(u_matrix, f_phi_matrix)
      u_optimizer.zero_grad()
      loss.backward()
      u_optimizer.step()
    u_matrix_list[i] = u_matrix.detach()
    if mode == 'train':
      for j in range(n_step):
        f_phi_matrix = agent.critic(z_phi_matrix)
        loss, loss_ctrl, neglogprior, loss_reg = loss_fn(u_matrix, f_phi_matrix)
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()




    print(f'iter {i}, loss: {loss.item():.4f}, loss_ctrl: {loss_ctrl.mean().item():.4f}, neglogprior: {neglogprior.item():.4f}, loss_reg: {loss_reg.item():.4f}')
    # fig, axis = plt.subplots(1, 1, figsize=(5, 5))

  f_phi_matrix = agent.critic(z_phi_matrix).detach()
  print('f_phi_matrix:', f_phi_matrix[0])

  root_filename = f'{initial_sample_idx}_{agent.feature_dim}to{phi_dim}_mle_sparse01_grw10'
  if mode == 'train':
    root_filename = f'{root_filename}_train1'
  elif mode == 'test':
    root_filename = f'{root_filename}_test'
  # plot_all_u(u_matrix_list, initial_u, save_path=f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/u_matrix_all.png')
  # plot_u(u_matrix, initial_u, f_phi_matrix, initial_f_phi_matrix, agent.feature_dim, save_path=f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/u_matrix.png')
  np.save(f'./kms/u_matrix_{root_filename}.npy', u_matrix.detach().cpu().numpy())
  # compare_action_ll(agent, initial_state, initial_action, initial_task, u_matrix, batch_size=sample_len)
  torch.save(agent.critic.state_dict(), f'./kms/critic_{root_filename}.pth')  
  # print('./kms/critic_16_map.pth')
  # agent.critic = agent.critic.to('cpu')
  # u_matrix = u_matrix.to('cpu')
  # average_state_ar, average_action_ar = collect_action_to_phi_all(args, dataset, agent, phi_dim)

  # pickle.dump({'average_state_ar': average_state_ar, 'average_action_ar': average_action_ar,
  #              'initial_state': initial_state.detach().cpu().numpy(), 'initial_task': initial_task.detach().cpu().numpy(),
  #              'u_matrix': u_matrix.detach().cpu().numpy()},
  #             open(f'./figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/fit_soft_info.pkl', 'wb'))
  # print(f'./figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/fit_soft_info.pkl')
  # average_state_ar = average_action_ar = None
  # print('u_matrix:', u_matrix.shape)
  
  # pair_gif_and_u(initial_state.detach().cpu().numpy(), u_matrix.detach().cpu().numpy(), initial_task.detach().cpu().numpy(),
  #                average_state_ar, average_action_ar,
  #                f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/video_{root_filename}.mp4',
  #                dpi=100)
  times = 1000
  auc_agents = np.zeros((times, ))
  auc_linears = np.zeros((times, ))
  for i in range(times):
    auc_agent, auc_linear = cal_plot_auc(initial_state, initial_action, initial_task, initial_u, 
                  u_matrix, dataset, agent, batch_size=sample_len, save_path=f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/auc_{root_filename}.pdf',
                  seed=i, device=device)
    auc_agents[i] = auc_agent
    auc_linears[i] = auc_linear
  print('auc_agent:', auc_agents.mean(), 'auc_linear:', auc_linears.mean())
  # fig, axis = plt.subplots(1, 1, figsize=(5, 5))
  # axis.errorbar(np.arange(1), auc_agents.mean(), yerr=auc_agents.std(), label='agent')
  # axis.errorbar(np.arange(1)+1, auc_linears.mean(), yerr=auc_linears.std(), label='linear')
  # axis.set_title(f'auc: {auc_agents.mean():.4f} +/- {auc_agents.std():.4f}, linear: {auc_linears.mean():.4f} +/- {auc_linears.std():.4f}')
  # axis.legend()
  # save_fig(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/auc_{root_filename}.png')
  return auc_agents.mean(), auc_agents.std(), auc_linears.mean(), auc_linears.std()





def fit_soft_syllable_batch(args, dataset, agent):
  finetune_agent = Finetune_Agent(args, dataset, agent)
  finetune_agent.fit_soft_syllable_batch()


def fit_train_test_opal(args, dataset, agent):

  np.random.seed(4)
  auc_all = open('./kms/auc_average.txt', 'r')
  auc_all = auc_all.readlines()
  train_auc_agent = []
  train_auc_linear = []
  test_auc_agent = []
  test_auc_linear = []
  train_idxs = []
  test_idxs = []
  for i in range(len(auc_all)):
      auc_all[i] = auc_all[i].split()
      print(auc_all[i])
      if 'train' in auc_all[i][0]:
          train_auc_agent.append(float(auc_all[i][1]))
          # print(auc_all[i][1])
          train_auc_linear.append(float(auc_all[i][2]))
          # print(auc_all[i][2])
          train_idxs.append(float(auc_all[i][0].split(':')[-1]))
      
      if 'test' in auc_all[i][0]:
          test_auc_agent.append(float(auc_all[i][1]))
          # print(auc_all[i][1])
          test_auc_linear.append(float(auc_all[i][2]))
          # print(auc_all[i][2])
          test_idxs.append(float(auc_all[i][0].split(':')[-1]))
  # print('train_auc_agent:', train_auc_agent)
  # print('train_auc_linear:', train_auc_linear)
  train_auc_agent = np.array(train_auc_agent)
  train_auc_linear = np.array(train_auc_linear)
  test_auc_agent = np.array(test_auc_agent)
  test_auc_linear = np.array(test_auc_linear)
  train_idxs = np.array(train_idxs).astype(np.int64)
  test_idxs = np.array(test_idxs).astype(np.int64)
  f_path = f'./kms/opal/auc_average.txt'
  f = open(f_path, 'w')
  train_auc_opal = np.zeros((len(train_auc_agent), ))
  test_auc_opal = np.zeros((len(train_auc_agent), ))

  for i in range(len(train_auc_agent)):
    train_idx = train_idxs[i]
    print('train_idx:', train_idx)
    auc_opal_mean, auc_opal_std = fit_latent_iql(args, dataset, agent, mode='train', \
                                  initial_sample_idx=train_idx, train_idx=None)
    train_auc_opal[i] = auc_opal_mean
    print('train_auc_agent:', train_auc_agent[i], 'train_auc_linear:', train_auc_linear[i],
          'train_auc_opal:', train_auc_opal[i])
    test_idx = test_idxs[i]
    print('test_idx:', test_idx)
    auc_opal_mean, auc_opal_std = fit_latent_iql(args, dataset, agent, mode='test', \
                                  initial_sample_idx=test_idx, train_idx=train_idx)
    test_auc_opal[i] = auc_opal_mean
    print('test_auc_agent:', test_auc_agent[i], 'test_auc_linear:', test_auc_linear[i],
          'test_auc_opal:', test_auc_opal[i])
    f.write(f'{i}\n train_idx:{train_idx} {train_auc_agent[i]} {train_auc_linear[i]} {train_auc_opal[i]}\n'
             f'test_idx:{test_idx} {test_auc_agent[i]} {test_auc_linear[i]} {test_auc_opal[i]}\n')
    f.flush()
  f.close()

def fit_latent_opal_z(args, dataset, agent, mode, initial_sample_idx, train_idx):
  device = 'cuda:0'

  sample_len = 250
  n_sample = 16
  # print('dataset device:', dataset.device)
  action_all = torch.zeros((n_sample, sample_len, agent.action_dim))
  for i in range(n_sample):
    sample_idx = np.random.randint(0, dataset.size-sample_len) + np.arange(sample_len)
    state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
    if i == 0:
      sample_idx = initial_sample_idx + np.arange(sample_len)
      state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
      task_onehot = F.one_hot(task.reshape(-1).long(), num_classes=agent.n_task).float()
      initial_state = state.clone().detach()
      initial_task = task.clone().detach()
      initial_action = action.clone().detach()
    action_all[i] = action

  z = agent.trajectory_latent(initial_state, initial_action)
  z = z.to(device).detach().requires_grad_()
  z_optimizer = torch.optim.Adam([z], lr=1e-3)
  
  iteration = 50
  n_step = 1000
  label = torch.zeros((sample_len, n_sample)).to(device)
  label[:,0] = 1
  label.requires_grad = False
  def loss_fn(z):
    actor_input = torch.concat([z, initial_state], -1)
    actor_dist = agent.actor(actor_input)
    log_prob = actor_dist.log_prob(action_all.to(device)).sum(-1)
    assert log_prob.shape == (n_sample, sample_len)
    loss_ctrl = nn.CrossEntropyLoss()(log_prob.T, label)

    neglogprior = (torch.diff(z, dim=0)**2).mean() * 10
    # neglogprior = torch.diag(u_matrix.T @ K_inv @ u_matrix).mean()*coef


    loss_reg = (z.abs()).mean() * 0.1
    # loss_reg = torch.zeros_like(neglogprior)
    loss = loss_ctrl + neglogprior + loss_reg
    return loss, loss_ctrl, neglogprior, loss_reg
  
  for i in range(iteration):
    for j in range(n_step):
      loss, loss_ctrl, neglogprior, loss_reg = loss_fn(z)
      z_optimizer.zero_grad()
      loss.backward()
      z_optimizer.step()

  times = 1000
  auc_news = np.zeros((times, ))
  for i in range(times):
    auc_new = cal_plot_auc_opal(state, action, z, dataset, agent, batch_size=sample_len,
                  seed=i, device=device)
    auc_news[i] = auc_new
  return auc_news.mean(), auc_news.std()

def cal_plot_auc_opal_z(state, action, z, dataset, agent, batch_size, seed, device):
  sample_idx = np.random.randint(0, dataset.size-batch_size)+np.arange(batch_size)
  state_2, action_2, next_state_2, reward_2, done_2, task_2, next_task_2 = unpack_batch(dataset.take(sample_idx))
  action_2 = action_2.to(device)
  pos_logll = agent.action_loglikelihood_z(state, action, z).detach().cpu().numpy()
  neg_logll = agent.action_loglikelihood_z(state, action_2, z).detach().cpu().numpy()
  y_agent_true = np.concatenate([np.ones_like(pos_logll), np.zeros_like(neg_logll)])
  auc_agent = roc_auc_score(y_agent_true, np.concatenate([pos_logll, neg_logll]))
  return auc_agent

def fit_latent_iql(args, dataset, agent, mode, initial_sample_idx, train_idx):
  # replay_buffer, state_dim, action_dim, n_task = load_all_keymoseq('test', args.dir, args.device)
  device = 'cuda:0'
  sample_len = 250
  sample_idx = initial_sample_idx + np.arange(sample_len)
  state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
  times = 1000
  auc_news = np.zeros((times, ))
  for i in range(times):
    auc_new = cal_plot_auc_iql(state, action, dataset, agent, batch_size=sample_len,
                  seed=i, device=device)
    auc_news[i] = auc_new

  return auc_news.mean(), auc_news.std()

def cal_plot_auc_iql(state, action, dataset, agent, batch_size, seed, device):
  sample_idx = np.random.randint(0, dataset.size-batch_size)+np.arange(batch_size)
  state_2, action_2, next_state_2, reward_2, done_2, task_2, next_task_2 = unpack_batch(dataset.take(sample_idx))
  action_2 = action_2.to(device)
  pos_logll = agent.action_loglikelihood(state, action).detach().cpu().numpy()
  neg_logll = agent.action_loglikelihood(state, action_2).detach().cpu().numpy()
  y_agent_true = np.concatenate([np.ones_like(pos_logll), np.zeros_like(neg_logll)])
  auc_agent = roc_auc_score(y_agent_true, np.concatenate([pos_logll, neg_logll]))
  return auc_agent


def fit_train_test_new(args, dataset, agent):
  fit_function = fit_latent_hilp
  np.random.seed(4)
  auc_all = open('./kms/iql/auc_average.txt', 'r')
  auc_all_readlines = auc_all.readlines()
  f_path = f'./kms/{args.alg}/auc_average.txt'
  f = open(f_path, 'w')
  for i in range(len(auc_all_readlines)):
      auc_all_split = auc_all_readlines[i].split()
      if 'train' in auc_all_split[0]:
          train_idx = int(auc_all_split[0].split(':')[-1])
          auc_new_mean, auc_new_std = fit_function(args, dataset, agent, mode='train', \
                              initial_sample_idx=train_idx, train_idx=None)
          f.write(auc_all_readlines[i].split('\n')[0] + f' {auc_new_mean}\n')
          f.flush()

      elif 'test' in auc_all_split[0]:
          test_idx = int(auc_all_split[0].split(':')[-1])
          auc_new_mean, auc_new_std = fit_function(args, dataset, agent, mode='test', \
                                initial_sample_idx=test_idx, train_idx=train_idx)
          f.write(auc_all_readlines[i].split('\n')[0] + f' {auc_new_mean}\n')
          f.flush()

      else:
          f.write(auc_all_readlines[i])
          f.flush()
  f.close()



def fit_latent_hilp(args, dataset, agent, mode, initial_sample_idx, train_idx):
  # replay_buffer, state_dim, action_dim, n_task = load_all_keymoseq('test', args.dir, args.device)
  device = 'cuda:0'

  sample_len = 250
  n_sample = 16
  state_all = torch.zeros((n_sample, sample_len, agent.state_dim))
  action_all = torch.zeros((n_sample, sample_len, agent.action_dim))

  sample_idx = initial_sample_idx + np.arange(sample_len)
  for i in range(n_sample):
    sample_idx = np.random.randint(0, dataset.size-sample_len) + np.arange(sample_len)
    state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
    if i == 0:
      sample_idx = initial_sample_idx + np.arange(sample_len)
      state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
      initial_state = state.clone().detach()
      initial_task = task.clone().detach()
      initial_action = action.clone().detach()

    action_all[i] = action
  state_all = initial_state.unsqueeze(0).repeat(n_sample,1,1).to(device)
  action_all = action_all.to(device)
  initial_z = agent.sample_z(sample_len).to(device)
  state = state.detach()
  action = action.detach()
  z_matrix = nn.Parameter(initial_z.clone().detach()).requires_grad_() # [sample_len, feature_dim]
  z_optimizer = torch.optim.Adam([z_matrix], lr=1e-4)
  agent.successor_net.eval()
  agent.successor_target_net.eval()

  iteration = 50
  n_step = 1000
  label = torch.zeros((sample_len, n_sample)).to(device)
  label[:,0] = 1
  label.requires_grad = False

  def loss_fn(z_matrix):
    z_matrix = F.normalize(z_matrix, dim=-1) * math.sqrt(agent.cfg.z_dim)
    F1, F2 = agent.successor_net(state_all, z_matrix.unsqueeze(0).repeat(n_sample,1,1), action_all)
    assert F1.shape == (n_sample, sample_len, agent.cfg.hidden_dim)
    assert F2.shape == (n_sample, sample_len, agent.cfg.hidden_dim)
    Q1 = torch.sum(F1 * z_matrix.unsqueeze(0), dim=-1)
    Q2 = torch.sum(F2 * z_matrix.unsqueeze(0), dim=-1)
    Q = (Q1 + Q2) / 2
    assert Q.shape == (n_sample, sample_len)
    loss_mle = nn.CrossEntropyLoss()(Q.T, label)

    neglogprior = (torch.diff(z_matrix, dim=0)**2).mean() * 10
    # neglogprior = torch.diag(u_matrix.T @ K_inv @ u_matrix).mean()*coef
    loss_reg = (z_matrix.abs()).mean() * 0.1
    # loss_reg = torch.zeros_like(neglogprior)
    loss = loss_mle + neglogprior + loss_reg
    return loss, loss_mle, neglogprior, loss_reg
  for i in range(iteration):
    
    for j in range(n_step):
      loss, loss_mle, neglogprior, loss_reg = loss_fn(z_matrix)
      z_optimizer.zero_grad()
      loss.backward()
      z_optimizer.step()
    #   z_matrix = F.normalize(z_matrix, dim=-1) * math.sqrt(agent.cfg.z_dim)
    #   with torch.no_grad():
    #     z_matrix.copy_(F.normalize(z_matrix, dim=-1) * math.sqrt(agent.cfg.z_dim))
    print(f'iter {i}, loss: {loss.item():.4f}, loss_mle: {loss_mle.mean().item():.4f}, neglogprior: {neglogprior.item():.4f}, loss_reg: {loss_reg.item():.4f}')

  root_filename = f'{initial_sample_idx}_{agent.cfg.hidden_dim}'
  if mode == 'train':
    root_filename = f'{root_filename}_train'
  elif mode == 'test':
    root_filename = f'{root_filename}_test'
  np.save(f'./kms/hilp/z_matrix_{root_filename}.npy', z_matrix.detach().cpu().numpy()) 
  times = 1000
  auc_agents = np.zeros((times, ))
  for i in range(times):
    auc_agent = cal_plot_auc_hilp(initial_state, initial_action, z_matrix, dataset, agent, batch_size=sample_len, save_path=f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/auc_{root_filename}.pdf',
                  seed=i, device=device)
    auc_agents[i] = auc_agent
  return auc_agents.mean(), auc_agents.std()

def cal_plot_auc_hilp(state, action, z_matrix, dataset, agent, batch_size, save_path, seed, device):
  sample_idx = np.random.randint(0, dataset.size-batch_size)+np.arange(batch_size)
  state_2, action_2, next_state_2, reward_2, done_2, task_2, next_task_2 = unpack_batch(dataset.take(sample_idx))
  action_2 = action_2.to(device)
  pos_logll = agent.action_loglikelihood(state, action, z_matrix).detach().cpu().numpy()
  neg_logll = agent.action_loglikelihood(state, action_2, z_matrix).detach().cpu().numpy()
  y_agent_true = np.concatenate([np.ones_like(pos_logll), np.zeros_like(neg_logll)])
  auc_agent = roc_auc_score(y_agent_true, np.concatenate([pos_logll, neg_logll]))
  return auc_agent