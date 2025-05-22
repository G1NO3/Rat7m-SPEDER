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
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib.gridspec import GridSpec
import moviepy.editor as mp
from matplotlib.animation import FFMpegWriter


def convert_gif_to_mp4(save_path):
  clip = mp.VideoFileClip(save_path)
  clip.write_videofile(save_path.replace('.gif', '.mp4'))

def save_fig(path, dpi=100):
  if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))
  plt.savefig(path)
  print(path)
  plt.close()

def get_edges(state_dim):
  if state_dim == 54:
    state_name = ['HeadF','HeadB','HeadL','SpineF','SpineM',
                  'SpineL','HipL','HipR','ElbowL','ArmL',
                  'ShoulderL','ShoulderR','ElbowR','ArmR','KneeR',
                  'KneeL','ShinL','ShinR']

    skeleton = [('HeadF', 'HeadB'), ('HeadF', 'HeadL'), ('HeadB', 'HeadL'),
                ('HeadB', 'SpineF'), ('HeadL', 'SpineF'), ('SpineF', 'SpineM'),
                ('SpineM', 'SpineL'), ('SpineF', 'ShoulderL'), ('ShoulderL', 'ElbowL'),
                ('ElbowL', 'ArmL'), ('SpineF', 'ShoulderR'), ('ShoulderR', 'ElbowR'),
                ('ElbowR', 'ArmR'), ('SpineM', 'HipL'), ('HipL', 'KneeL'),
                ('KneeL', 'ShinL'), ('SpineM', 'HipR'), ('HipR', 'KneeR'),
                ('KneeR', 'ShinR')]
    n_dim = 3
  elif state_dim == 16:
    state_name = ['spine4', 'spine3', 'spine2', 'spine1', 'head', 'nose', 'right ear', 'left ear']
    skeleton = [('spine4', 'spine3'), ('spine3', 'spine2'),
                ('spine2', 'spine1'), ('spine1', 'head'), ('head', 'nose'),
                ('head', 'left ear'), ('head', 'right ear')]
    n_dim = 2
  else:
    raise ValueError(f'state_dim {state_dim} not supported')
  edges = []
  for i in skeleton:
    edges.append((state_name.index(i[0]), state_name.index(i[1])))
  return edges, state_name, n_dim

def rasterize_figure(fig):
  canvas = fig.canvas
  canvas.draw()
  width, height = canvas.get_width_height()
  raster_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
  raster = raster_flat.reshape((height, width, 3))
  return raster

def plot_gif_onefig(stateseq, save_path, dpi=100):
  # stateseq: [timestep, state_dim]
  print(stateseq.shape)
  edges, state_name, n_dim = get_edges(stateseq.shape[-1])
  fig, axis = plt.subplots(1, 1, figsize=(5, 5))
  n_bodyparts = len(state_name)
  n_img = stateseq.shape[0]
  state_seqs_rs = stateseq.reshape(n_img, n_bodyparts, 2)
  print('state_seqs_to_plot:', state_seqs_rs.shape)
  print(state_seqs_rs[:,0])
  cmap = plt.cm.get_cmap('autumn')
  keypoint_colors = cmap(np.linspace(0, 1, len(state_name)))
  state_seqs_to_plot = state_seqs_rs - state_seqs_rs.mean(axis=(0,1), keepdims=True)
  # state_seqs_to_plot -= state_seqs_to_plot.mean(axis=(0,1), keepdims=True)
  axmin = -0.3
  axmax = 0.3
  aymin = -0.3
  aymax = 0.3
  # ymin = np.min(state_seqs_to_plot[...,1], axis=(-1,-2))
  # ymax = np.max(state_seqs_to_plot[...,1], axis=(-1,-2))
  # xmin = np.min(state_seqs_to_plot[...,0], axis=(-1,-2))
  # xmax = np.max(state_seqs_to_plot[...,0], axis=(-1,-2))
  # indicator = np.where((aymin > ymin) | (aymax < ymax) | (axmin > xmin) | (axmax < xmax), 1, 0)
  # aymin = np.where(indicator, -0.6, aymin)
  # aymax = np.where(indicator, 0.6, aymax)
  # axmin = np.where(indicator, -0.6, axmin)
  # axmax = np.where(indicator, 0.6, axmax)
  for i in range(n_img):
    for p1, p2 in edges:
      axis.plot(
          *state_seqs_to_plot[i, (p1, p2)].T,
          color=keypoint_colors[p1],
          linewidth=5.0,zorder=i*4)
    for p1, p2 in edges:
      axis.plot(
          *state_seqs_to_plot[i, (p1, p2)].T,
          color='black',
          linewidth=5.0*0.9,zorder=i*4+1)
    axis.scatter(
        *state_seqs_to_plot[i].T,
        c=keypoint_colors,
        edgecolors='black',
        s=100,zorder=i*4+2)
    # draw a white rectangle
    if i < n_img - 1:
        axis.fill_between([axmin, axmax], y1=aymin, y2=aymax, color='white', alpha=0.3,
                        zorder=i*4+3)
    axis.set_xlim(axmin, axmax)
    axis.set_ylim(aymin, aymax)
  # plt.show()
  save_fig(save_path, dpi=dpi)

def pair_gif_and_u(stateseq, u_matrix, taskseq, average_state_ar, average_action_ar, save_path, dpi):
  # fit_soft_info = pickle.load(open('./figure/kms/spedersac/A_f16_task78_ctrl_critic_nohidden/0/fit_soft_info.pkl', 'rb'))
  # stateseq = fit_soft_info['initial_state']
  # u_matrix = fit_soft_info['u_matrix']
  # taskseq = fit_soft_info['initial_task']
  # average_state_ar = fit_soft_info['average_state_ar']
  # average_action_ar = fit_soft_info['average_action_ar']
  # save_path = './figure/kms/spedersac/A_f16_task78_ctrl_critic_nohidden/0/pair_gif_and_u.gif'
  # stateseq: [timestep, state_dim]
  # u_matrix: [timestep, feature_dim]
  # taskseq: [timestep, 1]
  # average_state_ar: [feature_dim, state_dim]
  # average_action_ar: [feature_dim, action_dim]
  writer = FFMpegWriter(fps=15)

  edges, state_name, n_dim = get_edges(stateseq.shape[-1])
  n_bodyparts = len(state_name)
  timestep = stateseq.shape[0]
  cmap = plt.cm.get_cmap('viridis')
  keypoint_colors = cmap(np.linspace(0, 1, len(state_name)))
  state_seq_to_plot = stateseq.reshape(timestep, n_bodyparts, 2)
  state_seq_to_plot -= state_seq_to_plot.mean(axis=(0,1), keepdims=True)
  taskseq_to_plot = taskseq.reshape(-1)
  axmin = -0.2
  axmax = 0.2
  aymin = -0.2
  aymax = 0.2
  ymin = np.min(state_seq_to_plot[...,1], axis=(-1,-2))         
  ymax = np.max(state_seq_to_plot[...,1], axis=(-1,-2))
  xmin = np.min(state_seq_to_plot[...,0], axis=(-1,-2))
  xmax = np.max(state_seq_to_plot[...,0], axis=(-1,-2))   
  indicator = np.where((aymin > ymin) | (aymax < ymax) | (axmin > xmin) | (axmax < xmax), 1, 0)
  aymin = np.where(indicator, -0.5, aymin)
  aymax = np.where(indicator, 0.8, aymax)
  axmin = np.where(indicator, -0.7, axmin)
  axmax = np.where(indicator, 0.6, axmax) 
  fig_width, fig_height = 16, 4+3
  n_skill_shown = 6
  ax0_left = 0.8
  ax0_width = ax0_height = ax1_height = 3.2
  ax0_bottom = ax1_bottom = 0.48+3
  ax1_left = ax_kmslabel_left = 4.8
  ax1_width = ax_kmslabel_width = 10.88
  ax_kmslabel_bottom = 0.2+3
  ax_kmslabel_height = ax1_bottom - ax_kmslabel_bottom
  
  ax_skill_bottom = 0.3
  ax_skill_hspace = 0.2
  ax_gap_hspace = 0.3
  ax_skill_width = ax_skill_height = (fig_width - ax0_left*2 - ax_skill_hspace*(n_skill_shown-1))/n_skill_shown
  ax_skill_left = [ax0_left+(ax_skill_width+ax_skill_hspace)*i for i in range(n_skill_shown)]
  ax_skill_left = [x-ax_gap_hspace if i < n_skill_shown//2 else x+ax_gap_hspace for i, x in enumerate(ax_skill_left)]
  

  fig = plt.figure(figsize=(fig_width, fig_height))
  ax0 = fig.add_axes([ax0_left/fig_width, ax0_bottom/fig_height, ax0_width/fig_width, ax0_height/fig_height])
  ax1 = fig.add_axes([ax1_left/fig_width, ax1_bottom/fig_height, ax1_width/fig_width, ax1_height/fig_height])
  ax_kmslabel = fig.add_axes([ax_kmslabel_left/fig_width, ax_kmslabel_bottom/fig_height, ax_kmslabel_width/fig_width, ax_kmslabel_height/fig_height])
  ax_skill = [fig.add_axes([ax_skill_left[i]/fig_width, ax_skill_bottom/fig_height, ax_skill_width/fig_width, ax_skill_height/fig_height]) for i in range(n_skill_shown)]

  # set_ax_color_width(ax_skill[1], 'orange', 10)
  # set_ax_color_width(ax_skill[4], 'g', 10)
  ax_kmslabel.axis('off')
  ax0.axis('off')
  skill_cmap = plt.cm.get_cmap('Set1')
  rasters = []
  with writer.saving(fig, save_path, dpi=dpi):
    for i in range(timestep):
      ax0.clear()
      ax1.clear()
      ax_kmslabel.clear()
      [ax_skill[i].clear() for i in range(n_skill_shown)]
      ax0.set_xlim(axmin, axmax)
      ax0.set_ylim(aymin, aymax)
      ax1.set_xlim(0, u_matrix.shape[0])
      ax1.set_ylim(u_matrix.min(), u_matrix.max())
      ax_kmslabel.set_xlim(0, u_matrix.shape[0])
      ax_kmslabel.set_ylim(0, 1)
      for p1, p2 in edges:
        ax0.plot(
            *state_seq_to_plot[i, (p1, p2)].T,
            color=keypoint_colors[p1],
            linewidth=5.0,zorder=0)
      ax0.scatter(
          *state_seq_to_plot[i].T,
          c=keypoint_colors,
          s=100,zorder=1)
      u_matrix_idx = np.argsort(u_matrix, 1)
      skill_idx_to_plot = np.array([u_matrix_idx[i, -1], u_matrix_idx[i, -2], u_matrix_idx[i, -3],
                      u_matrix_idx[i, 2], u_matrix_idx[i, 1], u_matrix_idx[i, 0]])
      for j in range(u_matrix.shape[1]):
        y = u_matrix[:,j]
        # print((u_matrix_idx[:,-1]==j).shape)
        # y = np.where((u_matrix_idx[:, -1] == j) | (u_matrix_idx[:, -2] == j) | (u_matrix_idx[:, -3] == j), y, np.nan)
        ax1.plot(y, label=f'{j}', color=skill_cmap.colors[j%len(skill_cmap.colors)])
        # ax1.plot(u_matrix[:,j], label=f'{j}', color=skill_cmap.colors[j%len(skill_cmap.colors)])
      # for j in range(skill_idx_to_plot.shape[0]-3):
      #   ax1.plot(u_matrix[:, skill_idx_to_plot[j]], label=f'{skill_idx_to_plot[j]}', color=skill_cmap.colors[skill_idx_to_plot[j]%len(skill_cmap.colors)])
      
      ax1.vlines(i, ymin=u_matrix.min(), ymax=u_matrix.max(), color='black', linestyle='--')
      ax1.set_ylim(u_matrix.min(), u_matrix.max())
      ax1.legend(loc='upper right', fontsize=8)
      ax1.set_xticks([])
      ax1.set_title(f'{i}, first:{u_matrix_idx[i,-1], u_matrix_idx[i,-2], u_matrix_idx[i,-3]}, \
                    last:{u_matrix_idx[i,2], u_matrix_idx[i,1], u_matrix_idx[i,0]}')
      

      state_all = average_state_ar[skill_idx_to_plot]
      action_all = average_action_ar[skill_idx_to_plot]
      show_sa_all(ax_skill, state_all, action_all, skill_idx_to_plot, skill_cmap)
      ax_skill[1].set_xlabel('Highest', fontsize=15, labelpad=12)
      ax_skill[4].set_xlabel('Lowest', fontsize=15, labelpad=12)
      ax_kmslabel.imshow(taskseq_to_plot.reshape(1,-1), aspect='auto', cmap='Set1',
                        extent=[0, u_matrix.shape[0], 0, 1])
      ax_kmslabel.set_yticks([])
      ax_kmslabel.set_xticks([])
      ax_kmslabel.set_xlabel('keymoseq label')
      ax_kmslabel.vlines(i, ymin=0, ymax=1, color='black', linestyle='--')
      # mark the task on the axis
      task_transition = np.where(np.diff(taskseq_to_plot))[0]
      task_unique_order = taskseq_to_plot[task_transition+1]
      task_unique_order = np.concatenate(([taskseq_to_plot[0]], task_unique_order))
      task_transition = np.concatenate(([0], task_transition, [taskseq_to_plot.shape[0]-1]))
      for j in range(len(task_unique_order)):
        xcoord = (task_transition[j] + task_transition[j+1]) / 2 + 0.5
        ax_kmslabel.annotate(f'{int(task_unique_order[j])}', xy=(xcoord, 0.5),
                          fontsize=14, ha='center', va='center')
      # rasters.append(rasterize_figure(fig))
      writer.grab_frame()
  # writer.finish()
  plt.close(fig)
  print(save_path)
  # pil_images = [Image.fromarray(np.uint8(img)) for img in rasters]
  # Save the PIL Images as an animated GIF
  # if not os.path.exists(os.path.dirname(save_path)):
  #   os.makedirs(os.path.dirname(save_path))
  # pil_images[0].save(
  #     save_path,
  #     save_all=True,
  #     append_images=pil_images[1:],
  #     duration=100,
  #     loop=0,
  # )
  # print(save_path)
  # convert_gif_to_mp4(save_path)

def show_sa_all(axs, state_all, action_all, skill_all, skill_cmap):
  # state_all: [n_sample, state_dim]
  # action_all: [n_sample, action_dim]
  edges, state_name, n_dim = get_edges(state_all.shape[-1])
  n_bodyparts = len(state_name)
  n_sample = state_all.shape[0]
  state_seqs_to_plot = state_all.reshape(-1, n_bodyparts, 2)
  action_seqs_to_plot = action_all.reshape(-1, n_bodyparts, 2)
  cmap = plt.cm.get_cmap('autumn')
  keypoint_colors = cmap(np.linspace(0, 1, len(state_name)))
  axmin = -0.3
  axmax = 0.3
  aymin = -0.3
  aymax = 0.3
  xym = [axmin, axmax, aymin, aymax]
  skill_colors = [skill_cmap.colors[i%len(skill_cmap.colors)] for i in range(n_sample)]
  for i in range(n_sample):
    show_sa_single(axs[i], skill_all[i], state_seqs_to_plot[i], action_seqs_to_plot[i], 
                   edges, keypoint_colors, skill_colors[i], axspine_width=10, xym=xym)


def show_sa_single(ax, syllable, state, action, edges, keypoint_colors, syllable_color, axspine_width=10, xym=[-0.3,0.3,-0.3,0.3]):
  show_s(ax, state, edges, keypoint_colors)
  show_a(ax, state, action)
  ax.set_title(f'{syllable}', fontsize=20)
  ax.set_xlim(xym[0], xym[1])
  ax.set_ylim(xym[2], xym[3])
  ax.set_xticks([])
  ax.set_yticks([])
  set_ax_color_width(ax, syllable_color, axspine_width)

def show_s(ax, state_seq, edges, keypoint_colors):
  # state_seq: [n_bodyparts, 2]
  '''
      for p1, p2 in edges:
        axis.plot(
            *state_seqs_to_plot[i, (p1, p2)].T,
            color=keypoint_colors[p1],
            linewidth=5.0,zorder=i*4)
      for p1, p2 in edges:
        axis.plot(
            *state_seqs_to_plot[i, (p1, p2)].T,
            color='black',
            linewidth=5.0*0.9,zorder=i*4+1)
      axis.scatter(
          *state_seqs_to_plot[i].T,
          c=keypoint_colors,
          edgecolors='black',
          s=100,zorder=i*4+2)'''
  for p1, p2 in edges:
    ax.plot(
        *state_seq[(p1, p2), :].T,
        color=keypoint_colors[p1],
        linewidth=5.0, zorder=0)
  for p1, p2 in edges:
    ax.plot(
        *state_seq[(p1, p2), :].T,
        color='black',
        linewidth=5.0*0.9, zorder=1)
  ax.scatter(
      *state_seq.T,
      c=keypoint_colors,
      edgecolors='black',
      s=100, zorder=2)
def show_a(ax, state_seq, action_seq):
  n_bodyparts = state_seq.shape[0]
  for k in range(n_bodyparts):
    ax.quiver(state_seq[k, 0], state_seq[k, 1], 
              action_seq[k, 0], action_seq[k, 1], 
              angles='xy', scale_units='xy', scale=0.07, color='purple',
              width=0.01, headwidth=3, headlength=5, zorder=3)
def set_ax_color_width(ax, color, linewidth):
  # ax.axis('off')
  for spine in ax.spines.values():
    spine.set_color(color)
    spine.set_linewidth(linewidth)

def agent_likelihood_fn(agent, state, action, task, u_matrix):
  # state: [batch, state_dim]
  # action: [batch, action_dim]
  # u_matrix: [batch, feature_dim]
  f_phi = agent.critic(agent.phi(torch.concat([state, action], -1)))
  q = torch.sum(f_phi * u_matrix, dim=-1)
  # z_phi = agent.phi(torch.concat([state, action], -1))
  # q = torch.sum(z_phi * u_matrix, dim=-1)
  return q


def linear_loglikelihood(state, action, task, lr):
  # print('lr:', type(lr))
  lr_coef = torch.FloatTensor(lr['coef_matrix'])
  lr_intercept = torch.FloatTensor(lr['intercept_matrix'])
  lr_var = torch.FloatTensor(lr['var_matrix'])
  lr_covar = torch.FloatTensor(lr['covar_matrix'])
  lr_covar_inv = torch.FloatTensor(lr['covar_matrix_inv'])
  lr_logdet = torch.FloatTensor(lr['logdet_matrix'])
  # print('lr_coef:', lr_coef.shape, 'lr_intercept:', lr_intercept.shape, 'lr_var:', lr_var.shape)
  n_task = lr_coef.shape[0]
  batch_size = state.shape[0]
  state_dim = state.shape[-1]
  action_dim = action.shape[-1]
  assert lr_coef.shape == (n_task, state.shape[-1], action.shape[-1])
  assert lr_intercept.shape == (n_task, action.shape[-1],)
  assert lr_var.shape == (n_task, action.shape[-1],)
  task = task.squeeze(-1).long()
  linear_coef_matrix = lr_coef[task]
  linear_intercept_matrix = lr_intercept[task]
  # linear_var_matrix = lr_var[task]
  linear_covar_matrix = lr_covar[task]
  linear_covar_matrix_inv = lr_covar_inv[task]
  linear_logdet_matrix = lr_logdet[task]
  assert linear_coef_matrix.shape == (batch_size, action_dim, state_dim)
  assert linear_intercept_matrix.shape == (batch_size, action_dim)
  assert linear_covar_matrix.shape == (batch_size, action_dim, action_dim)
  assert state.shape == (batch_size, state_dim)
  # batch_size, n_action_dim, bins, action_dim, state_dim = linear_coef_matrix.shape
  # print('state:', state.shape, 'linear_coef_matrix:', linear_coef_matrix.shape)
  pred_action = torch.matmul(state.unsqueeze(-2), torch.transpose(linear_coef_matrix, -1, -2)).squeeze(-2) + linear_intercept_matrix
  err = pred_action - action
  quad = err.unsqueeze(-2) @ linear_covar_matrix_inv @ err.unsqueeze(-1)
  ll = - 0.5 * (quad.squeeze(-1) + linear_logdet_matrix)
  assert ll.shape == (batch_size, 1)
  # ll = - 0.5 * (torch.square(pred_action - action)) / linear_var_matrix
  # assert ll.shape == (batch_size, action_dim)
  # assert ll.shape == (batch_size, n_action_dim, bins, action_dim) 
  # return ll.sum(-1)
  return ll

def cal_plot_auc(state, action, task, initial_u, u_matrix, dataset, agent, batch_size, save_path, seed, device):
  agent.critic = agent.critic.to(device)
  agent.phi = agent.phi.to(device)
  initial_u = initial_u.to(device)
  u_matrix = u_matrix.to(device)
  state = state.to(device)
  action = action.to(device)
  task = task.to(device)
  sample_idx = np.random.randint(0, dataset.size-batch_size)+np.arange(batch_size)
  state_2, action_2, next_state_2, reward_2, done_2, task_2, next_task_2 = unpack_batch(dataset.take(sample_idx))
  action_2 = action_2.to(device)
  pos_logll = agent_likelihood_fn(agent, state, action, task, u_matrix).detach().cpu().numpy()
  neg_logll = agent_likelihood_fn(agent, state, action_2, task, u_matrix).detach().cpu().numpy()
  pos_logll_initial = agent_likelihood_fn(agent, state, action, task, initial_u).detach().cpu().numpy()
  neg_logll_initial = agent_likelihood_fn(agent, state, action_2, task, initial_u).detach().cpu().numpy()
  lr = pickle.load(open('./kms/linear_all.pkl', 'rb'))
  state = state.detach().cpu()
  action = action.detach().cpu()
  task = task.detach().cpu()
  action_2 = action_2.detach().cpu()
  positive_logll_linear = linear_loglikelihood(state, action, task, lr).detach().cpu().numpy()
  negative_logll_linear = linear_loglikelihood(state, action_2, task, lr).detach().cpu().numpy()
  auc_agent, auc_linear = plot_auc(pos_logll, neg_logll, positive_logll_linear, negative_logll_linear, 
                                   pos_logll_initial, neg_logll_initial, save_path)
  # fig, ax = plt.subplots(1, 2, figsize=(6, 3))
  # ax[0].hist(pos_logll, bins=20, alpha=0.6, density=True, color='orange')
  # ax[0].hist(neg_logll, bins=20, alpha=0.6, density=True, color='g')
  # ax[0].set_title(f'agent, auc={auc_agent:.4f}')
  # ax[1].hist(positive_logll_linear, bins=20, alpha=0.6, density=True, color='orange')
  # ax[1].hist(negative_logll_linear, bins=20, alpha=0.6, density=True, color='g')
  # ax[1].set_title(f'linear, auc={auc_linear:.4f}')
  # save_fig(save_path.replace('auc', 'hist'))
  return auc_agent, auc_linear

def plot_auc(positive_logll, negative_logll, pos_lr_logll, neg_lr_logll, initial_pos_logll, initial_neg_logll, save_path):
  y_agent_true = np.concatenate([np.ones_like(positive_logll), np.zeros_like(negative_logll)])
  y_lr_true = np.concatenate([np.ones_like(pos_lr_logll), np.zeros_like(neg_lr_logll)])
  # y_agent_initial = np.concatenate([np.ones_like(initial_pos_logll), np.zeros_like(initial_neg_logll)])
  auc_agent = roc_auc_score(y_agent_true, np.concatenate([positive_logll, negative_logll]))
  auc_lr = roc_auc_score(y_lr_true, np.concatenate([pos_lr_logll, neg_lr_logll]))
  # auc_agent_initial = roc_auc_score(y_agent_initial, np.concatenate([initial_pos_logll, initial_neg_logll]))
  # fig, ax = plt.subplots(1,1, figsize=(3.2,3))

  # fpr1, tpr1, _ = roc_curve(y_agent_true, np.concatenate([positive_logll, negative_logll]))
  # fpr2, tpr2, _ = roc_curve(y_lr_true, np.concatenate([pos_lr_logll, neg_lr_logll]))
  # fpr3, tpr3, _ = roc_curve(y_agent_initial, np.concatenate([initial_pos_logll, initial_neg_logll]))
  # ax.plot(fpr1, tpr1, color='orange', label=f'SKIL, AUC:{auc_agent:.4f}')
  # ax.plot(fpr2, tpr2, color='g', label=f'linear, AUC:{auc_lr:.4f}')
  # ax.plot(fpr3, tpr3, color='b', label=f'initial, AUC:{auc_agent_initial:.4f}')
  # ax.legend()
  # ax.plot([0, 1], [0, 1], color='k', linestyle='--')
  # ax.set_xlabel('False Positive Rate')
  # ax.set_ylabel('True Positive Rate')
  # plt.subplots_adjust(left=0.2, right=0.99, bottom=0.2, top=0.95)
  # save_fig(f'{save_path}', dpi=400)
  return auc_agent, auc_lr

def plot_u(u_matrix, initial_u, f_phi_matrix, initial_f_phi_matrix, feature_dim, save_path):
  initial_q = torch.sum(initial_f_phi_matrix[0] * initial_u, dim=-1).sum()
  optimized_q = torch.sum(f_phi_matrix[0] * u_matrix, dim=-1).sum()
  fig, ax = plt.subplots(2,1, figsize=(15,5))
  ax = ax.flatten()
  initial_u_numpy = initial_u.detach().cpu().numpy()
  u_matrix_numpy = u_matrix.detach().cpu().numpy()
  u_matrix_idx = np.argsort(u_matrix_numpy, 1)
  for j in range(feature_dim):
    ax[0].plot(initial_u_numpy[:,j], label=f'{j}')
  # ax[1].imshow(initial_f_phi_matrix[0].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
  for j in range(feature_dim):
    # ax[1].plot(u_matrix_numpy[:,j], label=f'{j}')
    y = u_matrix[:,j]
    # print((u_matrix_idx[:,-1]==j).shape)
    y = np.where((u_matrix_idx[:, -1] == j) | (u_matrix_idx[:, -2] == j) | (u_matrix_idx[:, -3] == j), y, np.nan)
    ax[1].plot(y, label=f'{j}', color=skill_cmap.colors[j%len(skill_cmap.colors)])
  # ax[3].imshow(f_phi_matrix[0].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
  cor = np.corrcoef(initial_u_numpy.flatten(), u_matrix_numpy.flatten())
  ax[0].legend()
  ax[1].legend()
  ax[0].set_title(f'initial u, q: {initial_q:.4f}') 
  ax[1].set_title(f'optimized u, q: {optimized_q:.4f}')
  # ax[1].set_title(f'initial f_phi, cor: {cor[0,1]:.4f}')
  print('initial u:', np.argmax(initial_u_numpy, axis=1))
  print('u_matrix:', np.argmax(u_matrix_numpy, axis=1))
  save_fig(save_path)
