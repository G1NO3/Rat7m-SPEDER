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
from main import load_rat7m, load_halfcheetah, load_keymoseq, load_keymoseq
from utils.util import unpack_batch
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from torch import nn, optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# import umap
import seaborn as sns
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel

def action_loglikelihood_multiple_syllables(args, dataset, agent):
  for i in range(agent.n_task):
    action_loglikelihood_single_syllable(args, dataset, agent, i)

def action_loglikelihood_single_syllable(args, dataset, agent, syllable, times=100):
  positive_logll = np.zeros(times)
  negative_logll = np.zeros(times)
  for i in range(times):
    while True:
      sample = dataset.sample(args.batch_size)
      task = sample.task
      # print(task)
      all_idx = torch.where(task == syllable)[0]
      if len(all_idx) > 0:
        break
    idx = all_idx[0]
    state = sample.state[all_idx]
    action = sample.action[all_idx]
    task = sample.task[all_idx]
    random_action = sample.action[torch.randint(0, len(all_idx), (len(all_idx),))]
    positive_logll[i] = agent.action_loglikelihood(state, action, task).detach().cpu().numpy()
    negative_logll[i] = agent.action_loglikelihood(state, random_action, task).detach().cpu().numpy()
  print('pos:', np.nanmean(positive_logll), np.nanstd(positive_logll))
  print('neg:', np.nanmean(negative_logll), np.nanstd(negative_logll))
  t, p = ttest_ind(positive_logll, negative_logll)
  print('t:', t, 'p:', p)
  fig, ax = plt.subplots(1,1, figsize=(10,10))
  ax.hist(positive_logll, bins=20, alpha=0.6, density=True, color='orange')
  ax.hist(negative_logll, bins=20, alpha=0.6, density=True, color='g')
  plt.legend(['positive sample', 'negative sample'])
  plt.title(f'action log likelihood, syllable {syllable}, p={p:.4f}')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/action_logll_{syllable}.png'
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  print(save_path)

def rollout_multiple_syllables(args, dataset, agent):
  for i in range(agent.n_task):
    rollout(args, dataset, agent, i)

def rollout(args, dataset, agent, syllable, timestep=2):
  # while True:
  #   sample = dataset.sample(args.batch_size)
  #   task = sample.task
  #   # print(task)
  #   all_idx = torch.where(task == syllable)[0]
  #   if len(all_idx) > 0:
  #     break
  idx = 354
  # state = dataset.state[idx:idx+1]
  # action = dataset.action[idx:idx+1]
  state = torch.FloatTensor(dataset.state[idx:idx+1])
  action = torch.FloatTensor(dataset.action[idx:idx+1])
  print('state:', state)
  print('action:', action)
  print('task:', dataset.task[idx:idx+1])
  stateseq = torch.zeros((timestep, *state.shape))
  actionseq = torch.zeros((timestep, *action.shape))
  stateseq[0] = state
  actionseq[0] = action
  for i in range(1, timestep):
    state, action, sp_likelihood, ap_q = agent.step(state, action, syllable)
    # print(i, 'action:', action)
    # print(sp_likelihood, ap_q)
    stateseq[i] = state
    actionseq[i] = action
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/rollout_{syllable}.gif'
  plot_gif(stateseq.squeeze(1), save_path)

def rollout_check_profile_all(args, dataset, agent, timestep=10):
  peak_score_ar = np.zeros((agent.n_task, ))
  higher_than_mean_ar = np.zeros((agent.n_task, ))
  higher_than_80quantile_ar = np.zeros((agent.n_task, ))
  action_to_linear_score_ar = np.zeros((agent.n_task, ))
  ap_q_ar = np.zeros((agent.n_task, ))
  stateseq_ar = np.zeros((agent.n_task, timestep, agent.state_dim))
  np.random.seed(4)
  for i in range(agent.n_task):
    peak_score, higher_than_mean, higher_than_80quantile, action_to_linear_score, ap_q, stateseq = \
      rollout_check_profile(args, dataset, agent, i, timestep, temperature=1, n=1000, step_size=1e-4)
    peak_score_ar[i] = peak_score
    higher_than_mean_ar[i] = higher_than_mean
    higher_than_80quantile_ar[i] = higher_than_80quantile
    action_to_linear_score_ar[i] = action_to_linear_score
    ap_q_ar[i] = ap_q
    stateseq_ar[i] = stateseq.squeeze(1)
  plot_gif_all_syllables(stateseq_ar, f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/rollout.gif')
  with open(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/rollout_profile.txt', 'w') as f:
    f.write('syllable, peak_score, higher_than_mean, higher_than_80quantile, action_to_linear_score, ap_q\n')
    for i in range(agent.n_task):
      f.write(f'{i}, {peak_score_ar[i]}, {higher_than_mean_ar[i]}, {higher_than_80quantile_ar[i]}, {action_to_linear_score_ar[i]}, {ap_q_ar[i]}\n')

def rollout_check_profile(args, dataset, agent, syllable, timestep, temperature, n, step_size):
  scale_factor = args.scale_factor
  torch.set_printoptions(threshold=torch.inf)
  # sample_idx = int(np.where(dataset.task == syllable)[0][0])
  all_idx = np.where(dataset.task == syllable)[0]
  sample_idx = int(all_idx[np.random.randint(0, len(all_idx))])
  sample_idx = 90
  print('sample_idx:', sample_idx, type(sample_idx))
  # sample_idx = 354
  # sample_idx = 161
  state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
  print('state:', state.shape, 'action:', action.shape)
  stateseq = torch.zeros((timestep, *state.shape))
  actionseq = torch.zeros((timestep, *action.shape))
  stateseq[0] = state
  actionseq[0] = action
  print(dataset.task[354:359])
  print('state:', dataset.state[354:359])  
  print('action:', dataset.action[354:359])
  lr = torch.load('./kms/linear_model.pth')
  batch_size = 1
  bins = 21
  peak_score_ar = np.zeros((timestep, ))
  higher_than_mean_ar = np.zeros((timestep, ))
  higher_than_80quantile_ar = np.zeros((timestep, ))
  action_to_linear_score_ar = np.zeros((timestep, ))
  ap_q_ar = np.zeros((timestep, ))
  for i in range(1,timestep):
    print(i, 'action:', action)
    action_pred = torch.FloatTensor(lr.predict(state.detach().cpu().numpy()))
    # action_pred_continuous = torch.FloatTensor(action_pred_continuous)

    assert next_state.shape == (batch_size, agent.state_dim)
    center = (bins-1)//2
    total_range = 20/scale_factor
    incremental_matrix = torch.eye(agent.action_dim).reshape(agent.action_dim, 1, agent.action_dim).repeat(1, bins, 1) \
                      * ((torch.arange(bins) - center) * total_range/(bins-1)).reshape(1, bins, 1)
    assert incremental_matrix.shape == (agent.action_dim, bins, agent.action_dim)
    new_action = action.reshape(batch_size, 1, 1, agent.action_dim) + incremental_matrix.reshape(1, agent.action_dim, bins, agent.action_dim)

    next_state = state + action
    next_state_batch = next_state.reshape(batch_size, 1, 1, agent.state_dim).repeat(1, agent.action_dim, bins, 1)
    # state_batch = state.reshape(batch_size, 1, 1, action_dim).repeat(1, agent.action_dim, bins, 1)
    task_batch = task.reshape(batch_size, 1, 1, 1).repeat(1, agent.action_dim, bins, 1)
    batch_q = agent.action_loglikelihood(next_state_batch, new_action, task_batch)[1].detach().cpu().squeeze(0)
    # print('batch_q:', batch_q.shape)
    assert batch_q.shape == (agent.action_dim, bins)
    peak_score, higher_than_mean, higher_than_80quantile, action_to_linear_score = draw_profile(agent, batch_q, new_action, state, action, action_pred, i, args.scale_factor)
    peak_score_ar[i] = peak_score
    higher_than_mean_ar[i] = higher_than_mean
    higher_than_80quantile_ar[i] = higher_than_80quantile
    action_to_linear_score_ar[i] = action_to_linear_score
    # state, action, sp_likelihood, ap_q = agent.step(state, action, syllable, temperature=0.1)
    # print('sprime ll:',sp_likelihood, 'q:',ap_q)
    # action = action_pred
    # state = state + (action_pred_continuous-2)/100##TODO: 
    next_state, next_action, sp_likelihood, ap_q = agent.step(state, action_pred, syllable, temperature=temperature, 
                                                              n=n, step_size=step_size)
    ap_q_ar[i] = ap_q
    state = next_state
    action = next_action
    stateseq[i] = state
    actionseq[i] = action
  # save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/rollout_{syllable}.gif'
  # plot_gif(stateseq.squeeze(1), save_path)
  return peak_score_ar.mean(), higher_than_mean_ar.mean(), higher_than_80quantile_ar.mean(), action_to_linear_score_ar.mean(), \
          ap_q_ar.mean(), stateseq

def draw_profile(agent, q, batch_action, state, action, action_pred, timestep, scale_factor):
  fig, axes = plt.subplots(4,4, figsize=(10,10))
  axes = axes.flatten()
  bins = q.shape[1]
  center = (bins-1)//2
  tick_length = 20/scale_factor/(bins-1)
  for i in range(agent.action_dim):
    q_onedim = q[i]
    # print('q_onedim:', q_onedim.min(), q_onedim.max())
    # print('batch_action:', batch_action.shape)  
    axes[i].plot(batch_action[0, i, :, i], q_onedim)
    axes[i].vlines(action[0, i], 0, q_onedim.max(), color='r', linestyle='--', label='real a')
    axes[i].vlines(action_pred[0, i], 0, q_onedim.max(), color='k', linestyle='--', label='pred a')
    axes[i].legend()
    peak_i = torch.argmax(q_onedim)
    peak_a = batch_action[0, i, peak_i, i]
    # higher_than_80quantile = torch.where(logll[center] - torch.quantile(logll, 0.8, dim=-1)>0, True, False)
    axes[i].set_title(f'peak {peak_a:.2f}')

  peak_idx = torch.argmax(q, dim=1)
  peak_score = torch.where(peak_idx==center, 1., 0.).mean()
  higher_than_mean = torch.where(q[:, (bins-1)//2] - q.mean(1)>0, 1., 0.).mean()
  quantile = torch.quantile(q, 0.8, dim=-1)
  higher_than_80quantile = torch.where(q[:, center] - quantile>0, 1., 0.).mean()
  linear_idx = (action_pred - action)/tick_length + center
  action_to_linear_score = torch.where(linear_idx==center, 1., 0.).mean()
  plt.suptitle(f'step {timestep}, peak score: {peak_score}, higher than mean: {higher_than_mean},than 80 quantile: {higher_than_80quantile},\n \
                action to linear score: {action_to_linear_score}')
  plt.tight_layout()
  fig_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/profile_ll_action_{timestep}.png'
  if not os.path.exists(os.path.dirname(fig_path)):
    os.makedirs(os.path.dirname(fig_path))
  plt.savefig(fig_path)
  print(fig_path)
  return peak_score, higher_than_mean, higher_than_80quantile, action_to_linear_score

def plot_gif_all_syllables(state_seqs_all, save_path):
  # state_seqs_all: [syllable, timestep, state_dim]
  edges, state_name, n_dim = get_edges(state_seqs_all.shape[2])
  fig, axis = plt.subplots(3, 4, figsize=(20, 15))
  n_bodyparts = len(state_name)
  n_img = state_seqs_all.shape[1]
  n_syllable = state_seqs_all.shape[0]
  dims, name = [0,1], 'xy'
  state_seqs_to_plot = state_seqs_all.reshape(-1, n_img, n_bodyparts, 2)
  cmap = plt.cm.get_cmap('viridis')
  keypoint_colors = cmap(np.linspace(0, 1, len(state_name)))
  rasters = []
  state_seqs_to_plot -= state_seqs_to_plot.mean(axis=(1,2), keepdims=True)
  axmin = -0.2
  axmax = 0.2
  aymin = -0.2
  aymax = 0.2

  ymin = np.min(state_seqs_to_plot[...,1], axis=(-1,-2))
  ymax = np.max(state_seqs_to_plot[...,1], axis=(-1,-2))
  xmin = np.min(state_seqs_to_plot[...,0], axis=(-1,-2))
  xmax = np.max(state_seqs_to_plot[...,0], axis=(-1,-2))
  print('ymin:', ymin.shape, 'ymax:', ymax.shape)
  indicator = np.where((aymin > ymin) | (aymax < ymax) | (axmin > xmin) | (axmax < xmax), 1, 0)
  aymin = np.where(indicator, -0.3, aymin)
  aymax = np.where(indicator, 0.3, aymax)
  axmin = np.where(indicator, -0.3, axmin)
  axmax = np.where(indicator, 0.3, axmax)
  for i in range(n_img):
    for j in range(n_syllable):
      axis[j//4, j%4].clear()
      for p1, p2 in edges:
        axis[j//4, j%4].plot(
            *state_seqs_to_plot[j, i, (p1, p2)].T,
            color=keypoint_colors[p1],
            linewidth=5.0)
      axis[j//4, j%4].scatter(
          *state_seqs_to_plot[j, i].T,
          c=keypoint_colors,
          s=100)
      axis[j//4, j%4].set_title(f'syllable {j}', fontsize=30)
      axis[j//4, j%4].axis('off')
      axis[j//4, j%4].set_xlim(axmin[j], axmax[j])
      axis[j//4, j%4].set_ylim(aymin[j], aymax[j])
    rasters.append(rasterize_figure(fig))
  pil_images = [Image.fromarray(np.uint8(img)) for img in rasters]
  # Save the PIL Images as an animated GIF
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  pil_images[0].save(
      save_path,
      save_all=True,
      append_images=pil_images[1:],
      duration=100,
      loop=0,
  )
  print(save_path)

def plot_gif(stateseq, save_path):
  # stateseq: [timestep, state_dim]
  clockwise, pc_to_plot = is_clockwise(stateseq)
  edges, state_name, n_dim = get_edges(stateseq.shape[1])
  fig, axis = plt.subplots(1, 1, figsize=(5, 6))
  n_bodyparts = len(state_name)
  n_img = stateseq.shape[0]
  if stateseq.shape[1] == 54:
    dims, name = [0,2], 'xz'
    state_seq_to_plot = stateseq.reshape(n_img, n_bodyparts, 3)[..., dims]
  else:
    dims, name = [0,1], 'xy'
    state_seq_to_plot = stateseq.reshape(n_img, n_bodyparts, 2)
    
  state_seq_to_plot -= state_seq_to_plot.mean(axis=(0,1), keepdims=True)
  cmap = plt.cm.get_cmap('viridis')
  keypoint_colors = cmap(np.linspace(0, 1, len(state_name)))
  rasters = []
  ymin = state_seq_to_plot[:,:,1].min()
  ymax = state_seq_to_plot[:,:,1].max()
  xmin = state_seq_to_plot[:,:,0].min()
  xmax = state_seq_to_plot[:,:,0].max()
  aymin=-0.2
  aymax=0.2
  axmin=-0.2
  axmax=0.2
  if aymin > ymin or aymax < ymax or axmin > xmin or axmax < xmax:
    aymin = -0.4
    aymax = 0.4
    axmin = -0.4
    axmax = 0.4
  for i in range(n_img):
    axis.clear()
    for p1, p2 in edges:
      axis.plot(
          *state_seq_to_plot[i, (p1, p2)].T,
          color=keypoint_colors[p1],
          linewidth=5.0,zorder=0)
    axis.scatter(
        *state_seq_to_plot[i].T,
        c=keypoint_colors,
        s=100,zorder=0)
    axis.quiver(0, 0, pc_to_plot[i,0], pc_to_plot[i,1], angles='xy', scale_units='xy', scale=10, color='r',
                zorder=1)
    axis.set_title(f'clockwise:{clockwise}', fontsize=30)
    axis.set_xlim(axmin, axmax)
    axis.set_ylim(aymin, aymax)
    
    rasters.append(rasterize_figure(fig))

  pil_images = [Image.fromarray(np.uint8(img)) for img in rasters]
  # Save the PIL Images as an animated GIF
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  pil_images[0].save(
      save_path,
      save_all=True,
      append_images=pil_images[1:],
      duration=100,
      loop=0,
  )
  print(save_path)

def rasterize_figure(fig):
  canvas = fig.canvas
  canvas.draw()
  width, height = canvas.get_width_height()
  raster_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
  raster = raster_flat.reshape((height, width, 3))
  return raster

def is_clockwise(stateseq):
  # stateseq: [timestep, state_dim]
  edges, state_name, n_dim = get_edges(stateseq.shape[1])
  head_idx = state_name.index('head')
  n_bodyparts = len(state_name)
  n_img = stateseq.shape[0]
  state_seq_to_plot = stateseq.reshape(n_img, n_bodyparts, 2)
  state_seq_to_plot = state_seq_to_plot - state_seq_to_plot.mean(axis=1, keepdims=True)
  _, _, vhs = np.linalg.svd(state_seq_to_plot)
  pc_to_plot = vhs[:,0]
  head_vector = state_seq_to_plot[:, head_idx, :]
  flip_sign = np.where(np.sum(head_vector*pc_to_plot, axis=-1, keepdims=True) < 0, -1, 1)
  pc_to_plot = pc_to_plot * flip_sign
  pc_seq_0 = pc_to_plot[:-1]
  pc_seq_1 = pc_to_plot[1:]
  cross_product = pc_seq_0[:,0]*pc_seq_1[:,1] - pc_seq_0[:,1]*pc_seq_1[:,0]
  # cross_product = state_seq_0[:,0]*state_seq_1[:,1] - state_seq_0[:,1]*state_seq_1[:,0]
  clockwise = np.sum(cross_product) < 0
  return clockwise, pc_to_plot

def plot_figure_PC(state, save_path):
  # state: [state_dim, ]
  edges, state_name, n_dim = get_edges(state.shape[-1])
  fig, axis = plt.subplots(1, 1, figsize=(5, 6))
  state_to_plot = state.reshape(-1, 2)
  state_to_plot -= state_to_plot.mean(axis=0)
  _, _, vhs = np.linalg.svd(state_to_plot)
  pc_to_plot = vhs[0]
  cmap = plt.cm.get_cmap('viridis')
  keypoint_colors = cmap(np.linspace(0, 1, len(state_name)))
  xmin = min(state_to_plot[:, 0].min(), pc_to_plot[0])
  xmax = max(state_to_plot[:, 0].max(), pc_to_plot[0])
  ymin = min(state_to_plot[:, 1].min(), pc_to_plot[1])
  ymax = max(state_to_plot[:, 1].max(), pc_to_plot[1])
  axis.set_xlim(xmin, xmax)
  axis.set_ylim(ymin, ymax)
  for p1, p2 in edges:
    axis.plot(
        *state_to_plot[(p1, p2),:].T,
        color=keypoint_colors[p1],
        linewidth=5.0)
  axis.scatter(
      *state_to_plot.T,
      c=keypoint_colors,
      s=100)
  axis.quiver(0, 0, pc_to_plot[0], pc_to_plot[1], angles='xy', scale_units='xy', scale=1, color='r')
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)

def check_action_space(args, dataset, agent):
  times = 100
  logll = np.zeros((times, ))
  for i in range(times):
    batch_1 = dataset.sample(args.batch_size)
    action_test = torch.rand((args.batch_size, agent.action_dim)).to('cuda:0') * 2 - 1
    logll[i] = agent.action_loglikelihood(batch_1.state, action_test, batch_1.task).detach().cpu().numpy()
  print(logll)
  fig, axis = plt.subplots(1,1, figsize=(10,10))
  axis.hist(logll, bins=20)
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/action_logll.png'
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  print(save_path)
  plt.close()

def visualize_wu(args, dataset, agent):
  plt.rcParams.update({'font.size': 15})
  n_task = agent.n_task
  z_w = agent.w(torch.eye(n_task).to('cuda:0')).detach().cpu().numpy()
  u1, u2 = agent.critic(torch.eye(n_task).to('cuda:0'))
  z_u1 = u1.detach().cpu().numpy()
  z_u2 = u2.detach().cpu().numpy()
  print('w:{}'.format(z_w.shape))
  sort_result = np.argsort(np.abs(z_w),axis=-1)[:,::-1]
  fig, axes = plt.subplots(3,1, figsize=(30,20))
  axes[0].imshow(z_w, cmap='coolwarm', aspect='auto')
  axes[0].set_title('w')
  axes[1].imshow(z_u1, cmap='coolwarm', aspect='auto')
  axes[1].set_title('u1')
  axes[2].imshow(z_u2, cmap='coolwarm', aspect='auto')
  axes[2].set_title('u2')
  fig.colorbar(axes[0].images[0], ax=axes[0])
  fig.colorbar(axes[1].images[0], ax=axes[1])
  fig.colorbar(axes[2].images[0], ax=axes[2])
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/wu.png'
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  print(save_path)
  fig, axes = plt.subplots(1,1, figsize=(30,10))
  axes.plot(z_w.T)
  axes.set_title('w')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/w.png'
  plt.subplots_adjust(left=0.1, right=0.9)
  plt.savefig(save_path)
  print(save_path)
  fig, axes = plt.subplots(1,1, figsize=(30,10))
  axes.plot(z_u1.T)
  axes.set_title('u1')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/u1.png'
  plt.subplots_adjust(left=0.1, right=0.9)
  plt.savefig(save_path)
  print(save_path)
  plt.close()
  n_cols = 10
  fig, axes = plt.subplots(n_task//n_cols+1, n_cols, figsize=(n_cols*10, (n_task//n_cols+1)*5))
  axes = axes.flatten()
  for i in range(n_task):
    axes[i].plot(z_w[i])
    axes[i].set_title(f'w{i},{sort_result[i,:6]}')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/w_line.png'
  plt.subplots_adjust(wspace=0.1,hspace=0.1,left=0.1, right=0.9)
  plt.tight_layout()
  plt.savefig(save_path)
  fig, axes = plt.subplots(n_task//n_cols+1, n_cols, figsize=(n_cols*10, (n_task//n_cols+1)*5))
  axes = axes.flatten()
  for i in range(n_task):
    axes[i].plot(z_u1[i])
    axes[i].set_title(f'u{i}')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/u1_line.png'
  plt.subplots_adjust(wspace=0.1,hspace=0.1,left=0.1, right=0.9)
  plt.tight_layout()
  plt.savefig(save_path)
  print(save_path)
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

def PCA_IG_skeleton(args, dataset, agent):
  # ig_matrix: [feature_dim, state_dim+action_dim]
  ig_matrix, ig_std_matrix, ig_matrix_agg_xyz, ig_std_matrix_agg_xyz, \
    state_name, action_name = cal_IG_matrix(args, dataset, agent, 3)
  print('ig_matrix_agg_xyz:', ig_matrix_agg_xyz)
  print('IG done')
  n_bodyparts = len(state_name)
  n_components = 10
  pca = PCA(n_components=n_components)
  assert ig_matrix_agg_xyz.shape == (agent.feature_dim, n_bodyparts*2)
  print(ig_matrix_agg_xyz.shape)  
  ig_pca = pca.fit_transform(ig_matrix_agg_xyz.T)
  # ig_pca = pca.components_
  print(ig_pca.shape)
  assert ig_pca.shape == (n_bodyparts*2, n_components)
  fig, ax = plt.subplots(1,1, figsize=(3, 3))
  ax.plot(pca.explained_variance_ratio_.cumsum(), marker='o', markersize=5)
  ax.set_title('PCA evr')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/pca_evr.png'
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  print(save_path)
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_pca.png'
  plot_IG_skeleton(ig_pca.T, state_name, n_components, save_path)

def draw_IG_skeleton(args, dataset, agent):
  edges, state_name, n_dim = get_edges(agent.state_dim)
  ig_matrix, ig_std_matrix, ig_matrix_agg_xyz, ig_std_matrix_agg_xyz = cal_IG_matrix(args, dataset, agent, 3)
  ig_matrix_xy = ig_matrix.reshape(agent.feature_dim, -1, n_dim)
  ig_matrix_x = ig_matrix_xy[:,:,0]
  ig_matrix_y = ig_matrix_xy[:,:,1]
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_skeleton_x.png'
  plot_IG_skeleton(ig_matrix_x, state_name, agent.feature_dim, save_path)
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_skeleton_y.png'
  plot_IG_skeleton(ig_matrix_y, state_name, agent.feature_dim, save_path)
  print('correlation:', np.corrcoef(ig_matrix_x.flatten(), ig_matrix_y.flatten()))


def plot_IG_skeleton(ig_matrix_agg_xyz, state_name, feature_dim, save_path):
  # ig_matrix: [feature_dim, state_dim+action_dim]
  edges, state_name, n_dim = get_edges(ig_matrix_agg_xyz.shape[1])
  col = 10*2
  row = feature_dim//(col//2) + 1
  fig, axes = plt.subplots(row, col, figsize=(col*5, row*6))
  axes = axes.flatten()
  ymean = np.load('./kms/s_mean.npy')
  print(ymean.shape)
  dims, name = [0,2], 'xz'
  if ymean.shape[1] == 54:
    ymean_to_plot = ymean.reshape(-1, 3)[:, :, dims]
  else:
    ymean_to_plot = ymean.reshape(-1, 2)
  cmap = plt.cm.get_cmap('viridis')
  keypoint_colors = cmap(np.linspace(0, 1, len(state_name)))
  n_bodyparts = len(state_name)
  assert ymean_to_plot.shape == (n_bodyparts, n_dim)
  assert ig_matrix_agg_xyz.shape == (feature_dim, 2*n_bodyparts)
  for i in range(feature_dim):
    for e in edges:
      # print(e, ymean_to_plot[e,:])
      axes[i*2].plot(
          *ymean_to_plot[e,:].T,
          color=keypoint_colors[e[0]],
          linewidth=5.0,
          zorder=0)
      axes[i*2+1].plot(
          *ymean_to_plot[e,:].T,
          color=keypoint_colors[e[0]],
          linewidth=5.0,
          zorder=0)
    node_colors = ['blue' if ig_matrix_agg_xyz[i, j] < 0 else 'red' for j in range(2*n_bodyparts)]
    axes[i*2].scatter(
          *ymean_to_plot.T,
          c=node_colors[:n_bodyparts],
          s=np.abs(ig_matrix_agg_xyz[i, :n_bodyparts])*300,
          zorder=1)
    axes[i*2+1].scatter(
          *ymean_to_plot.T,
          c=node_colors[n_bodyparts:],
          s=np.abs(ig_matrix_agg_xyz[i, n_bodyparts:])*300,
          zorder=1)

    axes[i*2].set_title(f'F{i} state', fontsize=30)
    axes[i*2+1].set_title(f'F{i} action', fontsize=30)
    axes[i*2].axis('off')
    axes[i*2+1].axis('off')
  plt.tight_layout()
  # plt.show()
  if save_path is not None:
    if not os.path.exists(os.path.dirname(save_path)):
      os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    print(save_path)
  return

def cal_IG_matrix(args, dataset, agent, times=3):
  edges, state_name, n_dim = get_edges(agent.state_dim)
  action_name = state_name
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  if os.path.exists(f'{save_path}/ig_matrix.npy'):
    print(f'load ig_matrix from {save_path}/ig_matrix.npy') 
    ig_matrix = np.load(f'{save_path}/ig_matrix.npy')
    ig_std_matrix = np.load(f'{save_path}/ig_std_matrix.npy')
    ig_matrix_agg_xyz = np.load(f'{save_path}/ig_matrix_agg.npy')
    ig_std_matrix_agg_xyz = np.load(f'{save_path}/ig_std_matrix_agg.npy')
    return ig_matrix, ig_std_matrix, ig_matrix_agg_xyz, ig_std_matrix_agg_xyz

  model = agent.phi
  ig = IntegratedGradients(model)
  # sa_ar = torch.zeros((args.times, args.batch_size, agent.state_dim+agent.action_dim))
  ig_matrix_all = torch.zeros((times, agent.feature_dim, agent.state_dim+agent.action_dim))
  ig_std_matrix_all = torch.zeros((times, agent.feature_dim, agent.state_dim+agent.action_dim))
  state_dataset = dataset.state
  action_dataset = dataset.action
  print(state_dataset.shape)
  state_min = state_dataset.min(0)
  print(state_min)
  state_max = state_dataset.max(0)
  print(state_max)
  action_min = action_dataset.min(0)
  print(action_min)
  action_max = action_dataset.max(0)
  print(action_max)
  for i in range(0, times):
    batch = dataset.sample(args.batch_size)
    state, action, next_state, reward, done, task, next_task = unpack_batch(batch)
    sa_ar = torch.cat([state, action], dim=1)
    sa_ar.requires_grad = True
    for j in range(agent.feature_dim):
      attr_ig, delta = ig.attribute(sa_ar, target=j, return_convergence_delta=True, 
                      baselines=-1)
      ig_matrix_all[i][j] = attr_ig.mean(dim=0).detach().cpu()# average over batch
      ig_std_matrix_all[i][j] = attr_ig.std(dim=0).detach().cpu()
    # use local gradients
    # phi_sa = agent.phi(sa_ar)
    # for j in range(agent.feature_dim):
    #   phi_sa_j = phi_sa[:, j].sum()
    #   phi_sa_j.backward(retain_graph=True)
    #   # print(sa_ar.grad.shape)
    #   ig_matrix_all[i][j] = sa_ar.grad.mean(dim=0).detach().cpu()
    #   ig_std_matrix_all[i][j] = sa_ar.grad.std(dim=0).detach().cpu()
    #   sa_ar.grad.zero_()
  # attr_ig, delta = ig.attribute(sa_ar_flatten, target=0, return_convergence_delta=True)
  # print(attr_ig.shape)
  # sns.heatmap(ig_matrix, cmap='coolwarm', ax=ax)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  print(agent.state_dim, len(np.arange(0,agent.state_dim,n_dim)))
  ig_matrix = ig_matrix_all.mean(dim=0)
  ig_std_matrix = ig_std_matrix_all.mean(dim=0)
  np.save(f'{save_path}/ig_matrix.npy', ig_matrix)
  np.save(f'{save_path}/ig_std_matrix.npy', ig_std_matrix)
  ig_matrix_agg_xyz = ig_matrix.reshape(agent.feature_dim, -1, n_dim).mean(dim=-1)
  ig_std_matrix_agg_xyz = ig_std_matrix.reshape(agent.feature_dim, -1, n_dim).mean(dim=-1)
  ig_matrix = ig_matrix.detach().cpu().numpy()
  ig_std_matrix = ig_std_matrix.detach().cpu().numpy()
  ig_matrix_agg_xyz = ig_matrix_agg_xyz.detach().cpu().numpy()
  ig_std_matrix_agg_xyz = ig_std_matrix_agg_xyz.detach().cpu().numpy()
  np.save(f'{save_path}/ig_matrix_agg.npy', ig_matrix_agg_xyz)
  np.save(f'{save_path}/ig_std_matrix_agg.npy', ig_std_matrix_agg_xyz)
  return ig_matrix, ig_std_matrix, ig_matrix_agg_xyz, ig_std_matrix_agg_xyz

def IntegratedGradients_attr(args, dataset, agent):
  fig, ax = plt.subplots(1,2, figsize=(20, 10))
  ig_matrix, ig_std_matrix, ig_matrix_agg_xyz, ig_std_matrix_agg_xyz, \
    state_name, action_name = cal_IG_matrix(args, dataset, agent)

  ax[0].imshow(ig_matrix[:,:agent.state_dim], cmap='coolwarm', aspect='auto')
  ax[0].set_title('state', fontsize=30)
  ax[0].set_xticks(np.arange(0,agent.state_dim,3), state_name, rotation=45)

  ax[1].imshow(ig_matrix[:,agent.state_dim:], cmap='coolwarm', aspect='auto')
  ax[1].set_title('action', fontsize=30)
  ax[1].set_xticks(np.arange(0,agent.action_dim,3), action_name, rotation=45)
  fig.colorbar(ax[0].images[0], ax=ax[0])
  fig.colorbar(ax[1].images[0], ax=ax[1])
  plt.subplots_adjust(wspace=0.1,hspace=0.1,left=0.1, right=0.9)
  plt.tight_layout()
  # for i in range(attr_ig.shape[0]):
    # ax.plot(attr_ig[i].detach().cpu().numpy())
  # ax.plot(attr_ig.mean(dim=0).detach().cpu().numpy())
  # plt.title('Integrated Gradients')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig.png'
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  plt.close()
  print(save_path)
  np.save(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_matrix.npy', ig_matrix)
  fig, ax = plt.subplots(1,2, figsize=(20, 10))
  ax[0].imshow(ig_std_matrix[:,:agent.state_dim], cmap='coolwarm', aspect='auto')
  ax[0].set_title('state', fontsize=30)
  ax[0].set_xticks(np.arange(0,agent.state_dim,3), state_name, rotation=45)
  ax[1].imshow(ig_std_matrix[:,agent.state_dim:], cmap='coolwarm', aspect='auto')
  ax[1].set_title('action', fontsize=30)
  ax[1].set_xticks(np.arange(0,agent.action_dim,3), action_name, rotation=45)
  fig.colorbar(ax[0].images[0], ax=ax[0])
  fig.colorbar(ax[1].images[0], ax=ax[1])
  plt.subplots_adjust(wspace=0.1,hspace=0.1,left=0.1, right=0.9)
  plt.tight_layout()
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_std.png'
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  plt.close()
  print(save_path)
  fig, ax = plt.subplots(16,8, figsize=(90,50))
  axes = ax.flatten()
  for i in range(agent.feature_dim):
    axes[2*i].plot(ig_matrix[i,:agent.state_dim])
    axes[2*i].set_xticks(np.arange(0,agent.state_dim,3), state_name, rotation=45)
    axes[2*i+1].plot(ig_matrix[i,agent.state_dim:])
    axes[2*i+1].set_xticks(np.arange(0,agent.action_dim,3), action_name, rotation=45)
    axes[2*i].set_title(f'F{i} state')
    axes[2*i+1].set_title(f'F{i} action')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_line.png'
  plt.subplots_adjust(wspace=0.1,hspace=0.1,left=0.1, right=0.9)
  plt.tight_layout()
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  print(save_path)
  # local_g_matrix = np.load(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/local_g_matrix.npy')
  # local_g_std_matrix = np.load(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/local_g_std_matrix.npy')
  # corr_g = np.corrcoef(local_g_matrix.flatten(), ig_matrix.flatten())
  # corr_g_std = np.corrcoef(local_g_std_matrix.flatten(), ig_std_matrix.flatten())
  # print('corr g:', corr_g)
  # print('corr g std:', corr_g_std)
  n_bodyparts = agent.state_dim // 3
  fig, axes = plt.subplots(1,2, figsize=(20,10))
  axes[0].imshow(ig_matrix_agg_xyz[:,:n_bodyparts], cmap='coolwarm', aspect='auto')
  axes[0].set_title('state', fontsize=30)
  axes[0].set_xticks(np.arange(0,n_bodyparts), state_name, rotation=45)
  axes[1].imshow(ig_matrix_agg_xyz[:,n_bodyparts:], cmap='coolwarm', aspect='auto')
  axes[1].set_title('action', fontsize=30)
  axes[1].set_xticks(np.arange(0,n_bodyparts), state_name, rotation=45)
  fig.colorbar(axes[0].images[0], ax=axes[0])
  fig.colorbar(axes[1].images[0], ax=axes[1])
  plt.subplots_adjust(wspace=0.1,hspace=0.1,left=0.1, right=0.9)
  plt.tight_layout()
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_agg.png'
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  print(save_path)
  np.save(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_matrix_agg.npy', ig_matrix_agg_xyz)
  # corr = np.corrcoef(ig_matrix_agg_xyz.flatten(), ig_matrix.flatten())
  # print('corr:', corr)
  fig, axes = plt.subplots(16,8, figsize=(90,50))
  axes = axes.flatten()

  for i in range(agent.feature_dim):
    axes[2*i].plot(ig_matrix_agg_xyz[i,:n_bodyparts])
    axes[2*i].set_xticks(np.arange(0,n_bodyparts), state_name[:n_bodyparts], rotation=45)
    axes[2*i+1].plot(ig_matrix_agg_xyz[i,n_bodyparts:])
    axes[2*i+1].set_xticks(np.arange(0,n_bodyparts), state_name[n_bodyparts:], rotation=45)
    axes[2*i].set_title(f'F{i} state')
    axes[2*i+1].set_title(f'F{i} action')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_line_agg.png'
  plt.subplots_adjust(wspace=0.1,hspace=0.1,left=0.1, right=0.9)
  plt.tight_layout()
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  print(save_path)



def cluster_in_phi_space(args, dataset, agent):
  state_dim = agent.state_dim
  action_dim = agent.action_dim
  feature_dim = agent.feature_dim
  phi_ar = torch.zeros((args.times, args.batch_size, feature_dim))
  label_ar = torch.zeros((args.times, args.batch_size))
  sa_ar = torch.zeros((args.times, args.batch_size, state_dim+action_dim))
  print('start clustering')
  for i in range(0, args.times):
    batch = dataset.sample(args.batch_size)
    state, action, next_state, reward, done = unpack_batch(batch)
    state.to('cuda:0')
    action.to('cuda:0')
    phi_ar[i] = agent.phi(torch.cat([state, action], dim=1))
    sa_ar[i] = torch.cat([state, action], dim=1)
    label_ar[i] = batch.task.flatten()
  print('clustering done')
  phi_ar_flatten = phi_ar.view(-1, feature_dim).detach().cpu().numpy()
  sa_ar_flatten = sa_ar.view(-1, state_dim+action_dim).detach().cpu().numpy()
  label_ar_flatten = label_ar.view(-1, feature_dim).detach().cpu().numpy()
  sa_phi_ar_flatten = np.concatenate((sa_ar_flatten, phi_ar_flatten), axis=1)
  print(phi_ar_flatten.shape)
  corr_matrix = np.corrcoef(sa_phi_ar_flatten.T)[:state_dim+action_dim, state_dim+action_dim:]
  print('corrcoef done')
  sns.heatmap(corr_matrix, cmap='coolwarm')
  print('heatmap done')
  # reducer = TSNE(n_components=2)
  # reducer = umap.UMAP(n_components=2)
  # reducer = PCA(n_components=2)
  # result = reducer.fit_transform(phi_ar_flatten)
  # plt.scatter(result[:, 0], result[:, 1], c=label_ar_flatten, cmap='Paired')
  # plt.show()
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/corrmatrix.png'
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  print(save_path)

def optimize_next_state(args, dataset, agent):
  feature_dim = agent.feature_dim
  state_dim, action_dim = agent.state_dim, agent.action_dim
  batch_size = 256
  focus_feature = 0
  # Define the target output
  # batch = dataset.sample(batch_size)
  batch = dataset.take(np.arange(10))
  state, action, next_state, reward, done, task, next_task = unpack_batch(batch)
  # Initialize the input with random values (ensure it has the right shape)
  # optimized_input = torch.zeros_like(next_state).requires_grad_(True)
  optimized_input = next_state.clone().detach().requires_grad_(True)

  # Define the optimizer to update the input 
  optimizer = optim.Adam([optimized_input], lr=0.01)

  # Optimization loop
  num_iterations = 1000  # Adjust based on convergence
  agent.phi.eval()
  # agent.mu.eval()
  z_phi = agent.phi(torch.cat([state, action], dim=-1)).detach()
  for i in range(num_iterations):

      # Forward pass
      z_mu_next = agent.mu(optimized_input)
      ll = torch.sum(z_phi * z_mu_next, dim=-1)
      # Compute the loss
      loss = -ll.mean()

      optimizer.zero_grad()
      # Backward pass
      loss.backward()

      # Update the input
      optimizer.step()

      # Optional: Clamp input to a specific range, e.g., [0, 1]
      # with torch.no_grad():
      #     optimized_input.clamp_(0, 1)

      # Print progress
      if i % 100 == 0:
          print(f"Iteration {i}, Loss: {loss.item()}")

  # The optimized input should now generate an output close to the target
  final_input = optimized_input.detach()
  print("Initial input:", next_state)
  print("Final optimized input:", final_input)
  print("error:", torch.norm(final_input - next_state, dim=-1))
  print('correlation:', torch.diag(torch.corrcoef(torch.cat([final_input, next_state], dim=-1).T)[state_dim:, :state_dim]))

def compare_bn_statistics(args, dataset, agent):
  # mu_original = agent.phi.state_dict()['trunk.2.running_mean']
  # var_original = agent.phi.state_dict()['trunk.2.running_var']
  # agent.phi.eval()
  # agent.mu.eval()
  # activation = []
  # def bn_input_hook(module, input, output):
      # input 是一个 tuple, 我们取第一个元素
  #     activation.append(input[0].detach())
  # print(agent.phi.state_dict().keys())
  # bn = agent.phi.get_submodule('trunk.2')
  # handle = bn.register_forward_hook(bn_input_hook)
  # z_phi = agent.phi(torch.cat([state, action], dim=-1)).detach()
  # handle.remove()
  # mu_new = activation[0].mean(0)
  # var_new = activation[0].var(0)
  # agent.phi.train()
  # z_phi = agent.phi(torch.cat([state, action], dim=-1)).detach()
  # print(torch.norm(mu_new * 0.1 + mu_original * 0.9-mu_updated), torch.norm(mu_original - mu_updated))
  # print(activation[0].mean(0))
  # print(agent.phi.state_dict()['trunk.2.running_mean'])
  # print(activation[0].var(0))
  # print(agent.phi.state_dict()['trunk.2.running_var'])
  # print('mean err:', torch.mean(torch.abs(activation[0].mean(0) - agent.phi.state_dict()['trunk.2.running_mean'])))
  # print('var err:', torch.mean(torch.abs((activation[0].var(0) - agent.phi.state_dict()['trunk.2.running_var']))))
  mu_original_train = np.load('mu_original.npy')
  var_original_train = np.load('var_original.npy')
  mu_new_train = np.load('mu_new.npy')
  var_new_train = np.load('var_new.npy')
  mu_updated_train = np.load('mu_updated.npy')
  var_updated_train = np.load('var_updated.npy')
  mu_original_test = np.load('mu_original2.npy')
  var_original_test = np.load('var_original2.npy')
  mu_new_test = np.load('mu_new2.npy')
  var_new_test = np.load('var_new2.npy')
  mu_updated_test = np.load('mu_updated2.npy')
  var_updated_test = np.load('var_updated2.npy')
  print(np.mean(np.abs(mu_original_train - mu_original_test)))  
  print(np.mean(np.abs(var_original_train - var_original_test)))
  print(np.mean(np.abs(mu_new_train - mu_new_test)))
  print(np.mean(np.abs(var_new_train - var_new_test)))
  print(np.mean(np.abs(mu_updated_train - mu_updated_test)))
  print(np.mean(np.abs(var_updated_train - var_updated_test)))
  print(np.mean(np.abs(mu_original_test * 0.9 + mu_new_test * 0.1 - mu_updated_train)))

def test_logll_smoothly(args, dataset, agent):
  scale_factor = args.scale_factor
  sample_idx = 5876
  state, action, expected_next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
  fig, axes = plt.subplots(1,5, figsize=(15,3))
  axes = axes.flatten()
  # fig, axes = plt.subplots(1, 1, figsize=(20,5))
  bins = 31
  show_idx = 0
  change_dim = 0
  # linspaces = torch.stack([torch.linspace(next_state[show_idx,i]-1, next_state[show_idx,i]+1, bins) for i in range(agent.state_dim)], 0).transpose(1,0)
  # expected_next_state = state + action
  state_i_scan = torch.linspace(expected_next_state[show_idx,change_dim]-30/scale_factor, expected_next_state[show_idx,change_dim]+30/scale_factor, bins)
  logll = torch.zeros((bins,))
  phi = torch.zeros((bins, agent.feature_dim))
  mu = torch.zeros((bins, agent.feature_dim))
  cmap = plt.cm.get_cmap('viridis')
  linecolors = cmap(np.linspace(0, 1, bins))
  # idx = np.array([128, 285, 377])
  # idx = np.array([126])
  # key_value_mu = np.zeros((bins, len(idx)))
  # key_value_phi = np.zeros((bins, len(idx)))
  for j in range(bins):
    new_next_state = expected_next_state.clone() 
    new_next_state[show_idx,change_dim] = state_i_scan[j]
    # new_next_state[show_idx, i+1] = state_i_1_scan[j]
    # print('next_state:', next_state[show_idx])
    # print('new_next_state:', new_next_state[show_idx])
    logll_one, phi_one, mu_one = agent.state_likelihood(state, action, new_next_state)
    logll[j] = logll_one[show_idx].detach().cpu()
    phi[j] = phi_one[show_idx].detach().cpu()
    mu[j] = mu_one[show_idx].detach().cpu()
    # key_value_mu[j] = mu_one[show_idx, idx].detach().cpu()
    # key_value_phi[j] = phi_one[show_idx, idx].detach().cpu()
    # print(logll[j])
  axes[0].plot(state_i_scan, logll)
  axes[0].set_title(f'logll, change dim {change_dim}')
  axes[0].vlines(expected_next_state[show_idx,change_dim], logll.min(), logll.max(), colors='r', linestyles='dashed')
  axes[0].vlines(state[show_idx,change_dim], logll.min(), logll.max(), colors='g', linestyles='dashed')
  # for i in range(bins):
  #   axes[1].plot(phi[i], c=linecolors[i], alpha=0.5)
  # axes[1].set_title(f'phi, change dim {change_dim}')
  idx = np.where(np.any(np.where(np.abs(mu) > 0.01, 1, 0),0))[0]
  print('idx:',idx)
  key_value_mu = mu[:,idx]
  key_value_phi = phi[:,idx]
  max_y = max(mu.max(), phi.max())*1.1
  min_y = min(mu.min(), phi.min())*1.1
  for i in range(bins):
    axes[1].plot(mu[i], c=linecolors[i], alpha=0.5)
    # print(i, np.where(np.abs(mu[i]) > 0.01)[0])
  axes[1].set_title(f'mu, change dim {change_dim}')  
  axes[1].set_ylim(min_y, max_y)
  for i in range(len(idx)):
    axes[2].plot(key_value_mu[:,i], label=idx[i])
  axes[2].set_title(f'mu key value, dim {idx}')
  axes[2].set_ylim(min_y, max_y)
  axes[2].legend()
  for i in range(bins):
    axes[3].plot(phi[i], c=linecolors[i], alpha=0.5)
    # print(i, np.where(np.abs(phi[i]) > 0.01)[0])
  axes[3].set_title(f'phi, change dim {change_dim}')
  axes[3].set_ylim(min_y, max_y)
  for i in range(len(idx)):
    axes[4].plot(key_value_phi[:,i], label=idx[i])
  axes[4].set_title(f'phi key value, dim {idx}')
  axes[4].set_ylim(min_y, max_y)
  axes[4].legend()
  fig_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/profile_ll_onedim.png'
  plt.tight_layout()
  plt.subplots_adjust(hspace=1,left=0.05, right=0.99)
  if not os.path.exists(os.path.dirname(fig_path)):
    os.makedirs(os.path.dirname(fig_path))
  plt.savefig(fig_path)
  print(fig_path)


def test_logll(args, dataset, agent, times=100):
  batch_size = 256
  positive_ll = np.zeros(times)
  negative_ll = np.zeros(times)
  few_hots_mu = np.zeros((times, batch_size))
  few_hots_phi = np.zeros((times, batch_size))
  for i in range(0, times):
    batch = dataset.sample(batch_size)
    batch_2 = dataset.sample(batch_size)
    state, action, next_state, reward, done, task, next_task = unpack_batch(batch)
    # print(state.shape, action.shape, next_state.shape)
    # s_random, a_random, ns_random, _, _, _, _ = unpack_batch(batch_2)
    # next_state_2 = next_state
    # next_state_2 = batch_2.state
    # next_state_2 = next_state + torch.randn_like(next_state)
    next_state_2 = state
    a_2 = action
    # a_2 = torch.randn_like(action)
    # a_2 = batch_2.action
    # print('real a:', action)
    # print('random a:', a_random)
    # print('action err:', torch.mean(torch.abs(action - a_random)/torch.abs(action)))
    # print('correlation:', torch.diag(torch.corrcoef(torch.cat([action, a_random], dim=-1).T)[:action.shape[-1], action.shape[-1]:]).mean())
    # positive_phi = agent.phi(torch.cat([state, action], dim=-1))
    # negative_phi = agent.phi(torch.cat([state, a_random], dim=-1))
    # pos_max_phi, pos_min_phi = torch.max(positive_phi, 0)[0], torch.min(positive_phi, 0)[0]
    # neg_max_phi, neg_min_phi = torch.max(negative_phi, 0)[0], torch.min(negative_phi, 0)[0]
    # print('pos_z_score:', ((pos_max_phi - positive_phi.mean(0))/positive_phi.std(0)).max(), ((pos_min_phi - positive_phi.mean(0))/positive_phi.std(0)).min())
    # print('neg_z_score:', ((neg_max_phi - negative_phi.mean(0))/negative_phi.std(0)).max(), ((neg_min_phi - negative_phi.mean(0))/negative_phi.std(0)).min())
    # print('positive phi:', positive_phi)
    # print('negative phi:', negative_phi)
    # print('relative err:', torch.mean(torch.abs(positive_phi - negative_phi)/torch.abs(positive_phi)))
    # print('correlation:', torch.diag(torch.corrcoef(torch.cat([positive_phi, negative_phi], dim=-1).T)[:feature_dim, feature_dim:]).mean())
    # mu_next = agent.mu(next_state)
    # pos_ll = torch.sum(positive_phi * mu_next, dim=-1).reshape(-1,1)
    # neg_ll = torch.sum(negative_phi * mu_next, dim=-1).reshape(-1,1)
    # print('positive sample:', pos_ll.flatten())
    # print('negative sample:', neg_ll.flatten())
    # print('relative err:', torch.mean(torch.abs(pos_ll - neg_ll)/torch.abs(pos_ll)))
    # print('correlation:', torch.corrcoef(torch.cat([pos_ll, neg_ll], dim=-1).T)[0,1])
    # print(action)
    # print(a_random)
    pos_ll_one, pos_phi, pos_mu = agent.state_likelihood(state, action, next_state)
    neg_ll_one, neg_phi, neg_mu = agent.state_likelihood(state, a_2, next_state_2)
    # print('pos_ll:', pos_ll_one.detach().cpu().numpy())
    # print('neg_ll:', neg_ll_one.detach().cpu().numpy())
    # print('pos mu:', pos_mu.detach().cpu().numpy()) 
    # print('neg mu:', neg_mu.detach().cpu().numpy())
    # fig, ax = plt.subplots()
    # print(pos_phi.shape)
    for j in range(pos_phi.shape[0]):
    #   ax.plot(pos_phi[j], label=j)
    #   print('few_hot:',np.where(np.abs(pos_phi[j]) > 0.01, 1, 0).sum())
      few_hots_phi[i,j] = np.where(np.abs(pos_phi[j]) > 0.01, 1, 0).sum()
    # plt.legend()
    # plt.title('phi')
    # save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/phi_{i}.png'
    # plt.savefig(save_path)
    # plt.close()
    # fig, ax = plt.subplots()
    for j in range(pos_mu.shape[0]):
      # ax.plot(pos_mu[j], label=j)
      # print('few_hot:',np.where(np.abs(pos_mu[j]) > 0.01, 1, 0).sum())
      few_hots_mu[i,j] = np.where(np.abs(pos_mu[j]) > 0.01, 1, 0).sum()
    # plt.legend()
    # plt.title('mu')
    # save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/mu_{i}.png'
    # plt.savefig(save_path)
    # plt.close()
    positive_ll[i] = pos_ll_one.mean().detach().cpu().numpy()
    negative_ll[i] = neg_ll_one.mean().detach().cpu().numpy()
    # print('positive sample:', positive_ll[i])
    # print('negative sample:', negative_ll[i])
    # dist = agent.likelihood_network(torch.cat([state, action], dim=-1))
    # print('state:', state[0])
    # print('action:', action[0])
    # print('arandom', a_random[0])
    # scale_back_state = state * agent.state_std + agent.state_mean
    # scale_back_action = action * agent.action_std + agent.action_mean
    # scale_back_a_random = a_random * agent.action_std + agent.action_mean
    # predict_next_state = scale_back_state + scale_back_action
    # scaled_next_state = (predict_next_state - agent.state_mean) / agent.state_std
    # predict_next_state_random = (scale_back_state + scale_back_a_random - agent.state_mean) / agent.state_std
    # print('state+action:', scaled_next_state[0])
    # print('state+random:', predict_next_state_random[0])
    # print('dist:', dist.loc[0])
    # print('next_state:', next_state[0])  
    # print('dist_scale:', dist.scale[0])
    # fig, ax = plt.subplots()
    # ax.plot(action[0], 'r', label='real')
    # ax.plot(a_random[0], 'b', label='random')
    # plt.legend()
    # plt.title(f'log likelihood, p={positive_ll[i]:.4f}, n={negative_ll[i]:.4f}')
    # save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/a_{i}.png'
    # plt.savefig(save_path)
  # print(positive_ll)
  # print(negative_ll)
  # significance test
  t_stat, p_value = ttest_ind(positive_ll, negative_ll, equal_var=False)
  fig, ax = plt.subplots(figsize=(5,5))
  print('positive sample:', positive_ll.mean(), positive_ll.std())
  print('negative sample:', negative_ll.mean(), negative_ll.std())
  print('p_value:', p_value)
  # ax.bar(['positive', 'negative'], [positive_ll.mean(), negative_ll.mean()])
  ax.errorbar(['positive'], [positive_ll.mean()], yerr=[positive_ll.std()], fmt='o', capsize=5)
  ax.errorbar(['negative'], [negative_ll.mean()], yerr=[negative_ll.std()], fmt='o', capsize=5)
  # ax.hist(positive_ll, bins=2, density=True, alpha=0.6, color='orange')
  # ax.hist(negative_ll, bins=2, density=True, alpha=0.6, color='g')
  plt.legend(['positive sample', 'negative sample'])
  plt.title(f'likelihood, p={p_value}')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  plt.savefig(f'{save_path}/ll.png')
  print(f'{save_path}/ll.png')
  plt.close()
  fig, axes = plt.subplots(1,2, figsize=(10,5))
  axes[0].hist(few_hots_phi.flatten(), bins=20, density=True, alpha=0.6, color='orange')
  axes[0].set_title(f'phi hot, mean:{few_hots_phi.mean()}')
  axes[1].hist(few_hots_mu.flatten(), bins=20, density=True, alpha=0.6, color='orange')
  axes[1].set_title(f'mu hot, mean:{few_hots_mu.mean()}')
  plt.savefig(f'{save_path}/few_hots.png')
  print(f'{save_path}/few_hots.png')
  plt.close()
  return positive_ll.mean(), positive_ll.std(), negative_ll.mean(), negative_ll.std()

def action_test_logll(args, dataset, agent):
  scale_factor = args.scale_factor
  batch_size = args.batch_size
  times = 100
  positive_logll = np.zeros(times)
  positive_q = np.zeros(times)
  negative_logll = np.zeros(times)
  negative_q = np.zeros(times)
  lr_logll = np.zeros(times)
  lr_q = np.zeros(times)
  lr = torch.load('./kms/linear_model.pth')
  for i in range(times):
    batch_1 = dataset.sample(batch_size)
    batch_2 = dataset.sample(batch_size)
    # action_1 = torch.FloatTensor(lr.predict(batch_1.state.detach().cpu().numpy()))
    action_1 = batch_1.action
    action_2 = batch_2.action
    action_pred = torch.FloatTensor(lr.predict(batch_1.state.detach().cpu().numpy()))
    # action_2 = torch.randn_like(batch_1.action)
    print('original action:', batch_1.action[0], 'new action:', action_2[0]) 
    positive_logll_one, positive_q_one = agent.action_loglikelihood(batch_1.state, action_1, batch_1.task)
    positive_logll[i] = positive_logll_one.detach().cpu().numpy().mean()
    positive_q[i] = positive_q_one.detach().cpu().numpy().mean()
    negative_logll_one, negative_q_one = agent.action_loglikelihood(batch_1.state, action_2, batch_1.task)
    negative_logll[i] = negative_logll_one.detach().cpu().numpy().mean()
    negative_q[i] = negative_q_one.detach().cpu().numpy().mean()
    lr_logll_one, lr_q_one = agent.action_loglikelihood(batch_1.state, action_pred, batch_1.task)
    lr_logll[i] = lr_logll_one.detach().cpu().numpy().mean()
    lr_q[i] = lr_q_one.detach().cpu().numpy().mean()

    print('pos:', positive_logll[i], positive_q[i])
    print('neg:', negative_logll[i], negative_q[i])
    print('lr:', lr_logll[i], lr_q[i])
  
  print('pos:', np.mean(positive_logll), np.std(positive_logll), np.mean(positive_q), np.std(positive_q))
  print('neg:', np.mean(negative_logll), np.std(negative_logll), np.mean(negative_q), np.std(negative_q))
  print('lr:', np.mean(lr_logll), np.std(lr_logll), np.mean(lr_q), np.std(lr_q))
  t, p = ttest_ind(positive_logll, negative_logll)
  print('t:', t, 'p:', p)
  t2, p2 = ttest_ind(positive_logll, lr_logll)
  print('t2:', t2, 'p2:', p2)
  fig, ax = plt.subplots(1,2, figsize=(10,5))
  ax[0].hist(positive_logll, bins=20, alpha=0.6, density=True, color='orange')
  ax[0].hist(negative_logll, bins=20, alpha=0.6, density=True, color='g')
  ax[0].hist(lr_logll, bins=20, alpha=0.6, density=True, color='b')
  ax[0].legend(['positive sample', 'negative sample', 'lr pred'])
  ax[0].set_title(f'action log likelihood, p={p},\n p2={p2}')
  t,p = ttest_ind(positive_q, negative_q)
  print('t:', t, 'p:', p)
  t2,p2 = ttest_ind(positive_q, lr_q)
  print('t2:', t2, 'p2:', p2)
  ax[1].hist(positive_q, bins=20, alpha=0.6, density=True, color='orange')
  ax[1].hist(negative_q, bins=20, alpha=0.6, density=True, color='g')
  ax[1].hist(lr_q, bins=20, alpha=0.6, density=True, color='b')
  ax[1].legend(['positive sample', 'negative sample', 'lr pred'])
  ax[1].set_title(f'action q value, p={p},\n p2={p2}')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/action_logll.png'
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  print(save_path)
  plt.close()




def action_profile_likelihood(args, dataset, agent):
  scale_factor = args.scale_factor
  torch.set_printoptions(threshold=torch.inf)
  sample_idx = 1565
  state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
  fig, axes = plt.subplots(4,4, figsize=(10,10))
  axes = axes.flatten()
  bins = 31
  all_logll = torch.zeros(agent.action_dim, bins)
  show_idx = 0
  total_range = 100/scale_factor
  center = (bins-1)//2
  lr = torch.load('./kms/linear_model.pth')
  action_pred = lr.predict(state.detach().cpu().numpy())
  for i in range(agent.action_dim):
    action_i_scan = torch.linspace(action[show_idx,i]-total_range/2, action[show_idx,i]+total_range/2, bins)
    logll = torch.zeros((bins,))
    for j in range(bins):
      new_action = action.clone()
      new_action[show_idx,i] = action_i_scan[j]
      # new_next_state[show_idx, i+1] = state_i_1_scan[j]
      # print('next_state:', next_state[show_idx])
      # print('new_next_state:', new_next_state[show_idx])/
      # f.write(f'{new_next_state[show_idx]}\n')
      # f.write(f'{state[0]}\n')
      # f.write(f'{action[0]}\n')
      logll[j] = agent.action_loglikelihood(state, new_action, task)[1].detach().cpu()
      # f.write(f'{logll[j]}\n')
      # print(logll[j])
    all_logll[i] = logll
    axes[i].plot(action_i_scan, logll)
    # print(torch.exp(logll))
    # print(logll)
    # expect_next_state = state + action
    axes[i].vlines(action[show_idx, i], logll.min(), logll.max(), color='r', linestyle='--', label='real a')
    axes[i].vlines(action_pred[show_idx, i], logll.min(), logll.max(), color='k', linestyle='--', label='pred a')
    axes[i].legend()
    peak_i = torch.argmax(logll)
    higher_than_80quantile = torch.where(logll[center] - torch.quantile(logll, 0.8, dim=-1)>0, True, False)
    axes[i].set_title(f'peak {peak_i} higher than 80 {higher_than_80quantile}')
    # all_logll[i] = logll
    # print(next_state[show_idx])
  # f.close()
  peak_idx = torch.argmax(all_logll, dim=1)
  peak_score = torch.where(peak_idx==center, 1., 0.).mean()
  higher_than_mean = torch.where(all_logll[:, (bins-1)//2] - all_logll.mean(1)>0, 1., 0.).mean()
  quantile = torch.quantile(all_logll, 0.8, dim=-1)
  higher_than_80quantile = torch.where(all_logll[:, center] - quantile>0, 1., 0.).mean()
  plt.suptitle(f'sample {sample_idx}, peak score: {peak_score}, higher than mean: {higher_than_mean}, higher than 80 quantile: {higher_than_80quantile}')
  plt.tight_layout()
  fig_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/profile_ll_action.png'
  if not os.path.exists(os.path.dirname(fig_path)):
    os.makedirs(os.path.dirname(fig_path))
  plt.savefig(fig_path)
  print(fig_path)
  return

def action_profile_likelihood_batch(args, dataset, agent):
  torch.set_printoptions(threshold=torch.inf)
  scale_factor = args.scale_factor
  batch_size = 256
  times = 10
  bins = 21
  metric_matrix = np.zeros((times, 3))
  linear_model_metric_matrix = np.zeros((times, 3))
  linear_model = torch.load('./kms/linear_model.pth')
  action_dim = agent.action_dim
  for i in range(times):
    # sample_idxs = np.random.randint(0, dataset.size, batch_size)
    # print('sample_idxs:', sample_idxs)
    state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.sample(batch_size))
    # print('state:', state, 'action:', action, 'next_state:', next_state)
    # assert torch.isclose(state + action, next_state, atol=1e-6).all()
    assert next_state.shape == (batch_size, agent.state_dim)
    center = (bins-1)//2
    total_range = 20/scale_factor
    incremental_matrix = torch.eye(agent.action_dim).reshape(agent.action_dim, 1, agent.action_dim).repeat(1, bins, 1) \
                      * ((torch.arange(bins) - center) * total_range/(bins-1)).reshape(1, bins, 1)
    assert incremental_matrix.shape == (agent.action_dim, bins, agent.action_dim)
    new_action = action.reshape(batch_size, 1, 1, agent.action_dim) + incremental_matrix.reshape(1, agent.action_dim, bins, agent.action_dim)
    state_batch = state.reshape(batch_size, 1, 1, action_dim).repeat(1, agent.action_dim, bins, 1)
    task_batch = task.reshape(batch_size, 1, 1, 1).repeat(1, agent.action_dim, bins, 1)
    logll = agent.action_loglikelihood(state_batch, new_action, task_batch)[1].detach().cpu()
    print('logll:', logll.shape)
    assert logll.shape == (batch_size, agent.action_dim, bins)
    peak_idx = torch.argmax(logll, dim=-1)
    peak_score = torch.where(peak_idx==center, 1., 0.)
    peak_score = peak_score.mean()
    higher_than_mean = torch.where(logll[..., center] - logll.mean(-1)>0, 1., 0.).mean()
    quantile = torch.quantile(logll, 0.80, dim=-1)
    higher_than_80quantile = torch.where(logll[..., center] - quantile>0, 1., 0.).mean()
    print(f'peak score: {peak_score}, higher than mean: {higher_than_mean}, higher than 80 quantile: {higher_than_80quantile}')
    metric_matrix[i] = peak_score, higher_than_mean, higher_than_80quantile
    # linear model
    action_pred = linear_model.predict(state.detach().cpu().numpy())
    action_pred = torch.FloatTensor(action_pred)
    assert action_pred.shape == action.shape
    linear_model_logll = agent.action_loglikelihood(state, action_pred, task)[1].detach().cpu()
    # print('linear model logll:', linear_model_logll.shape, 'logll:', logll.shape, 'quantile:', quantile.shape)
    assert linear_model_logll.shape == (batch_size, )
    # print('linear model logll:', linear_model_logll)
    # print('logll mean:', logll.mean(-1).mean(-1))
    # print('quantile:', quantile.mean(-1))
    # print('center:', center)
    # print('action_pred:', action_pred)
    # print('action:', action)
    # print('action_pred - action:', (action_pred - action))
    action_pred_idx = torch.round((action_pred - action)/(total_range/(bins-1)))+center
    # print('action pred idx:', action_pred_idx)
    peak_score_linear = torch.where((action_pred_idx==peak_idx), 1., 0.).mean()
    # print('peak score linear:', peak_score_linear)
    higher_than_mean_linear = torch.where(linear_model_logll - logll.mean(-1).mean(-1)>0, 1., 0.).mean()
    # print('if higher than mean:', (linear_model_logll - logll.mean(-1).mean(-1)>0))
    higher_than_80quantile_linear = torch.where(linear_model_logll - quantile.mean(-1)>0, 1., 0.).mean()
    linear_model_metric_matrix[i] = peak_score_linear, higher_than_mean_linear, higher_than_80quantile_linear
  # print('metric:', metric_matrix.mean(0), metric_matrix.std(0)) 
  metric_mean = metric_matrix.mean(0)
  metric_std = metric_matrix.std(0)
  linear_metric_mean = linear_model_metric_matrix.mean(0) 
  linear_metric_std = linear_model_metric_matrix.std(0)
  text_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/action_metric.txt'
  if not os.path.exists(os.path.dirname(text_path)):
    os.makedirs(os.path.dirname(text_path))
  with open(text_path, 'w') as f:
    f.write(f'total range: {total_range}, bins: {bins}, batch size: {batch_size}, times: {times}\n')
    f.write(f'peak score: {metric_mean[0]:.4f} +- {metric_std[0]:.4f}\n')
    f.write(f'higher than mean: {metric_mean[1]:.4f} +- {metric_std[1]:.4f}\n')
    f.write(f'higher than 80 quantile: {metric_mean[2]:.4f} +- {metric_std[2]:.4f}\n')
    f.write(f'linear model peak score: {linear_metric_mean[0]:.4f} +- {linear_metric_std[0]:.4f}\n')
    f.write(f'linear model higher than mean: {linear_metric_mean[1]:.4f} +- {linear_metric_std[1]:.4f}\n')
    f.write(f'linear model higher than 80 quantile: {linear_metric_mean[2]:.4f} +- {linear_metric_std[2]:.4f}\n')
  print(text_path)
  return

def action_profile_likelihood_discrete(args, dataset, agent):
  scale_factor = args.scale_factor
  torch.set_printoptions(threshold=torch.inf)
  sample_idx = 587
  state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
  fig, axes = plt.subplots(4,4, figsize=(10,10))
  axes = axes.flatten()
  all_logll = torch.zeros(agent.action_dim//agent.n_action, agent.n_action)
  show_idx = 0
  lr = torch.load('./kms/linear_model_discrete.pth')
  action_pred = lr.predict(state.detach().cpu().numpy())
  action_pred = torch.FloatTensor(action_pred)
  action_pred = torch.clip(action_pred, 0, agent.n_action-1).long()
  for i in range(agent.action_dim//agent.n_action):
    logll = torch.zeros((agent.n_action,))
    for j in range(agent.n_action):
      new_action = action.clone()
      new_action[show_idx,i] = j
      # new_next_state[show_idx, i+1] = state_i_1_scan[j]
      # print('next_state:', next_state[show_idx])
      # print('new_next_state:', new_next_state[show_idx])/
      # f.write(f'{new_next_state[show_idx]}\n')
      # f.write(f'{state[0]}\n')
      # f.write(f'{action[0]}\n')
      logll[j] = agent.action_loglikelihood(state, new_action, task)[0].detach().cpu()

      # f.write(f'{logll[j]}\n')
      # print(logll[j])
    all_logll[i] = logll
    axes[i].bar(torch.arange(agent.n_action), logll)
    axes[i].vlines(action[show_idx, i], logll.min(), logll.max(), color='r', linestyle='--', label='real a')
    axes[i].vlines(action_pred[show_idx, i], logll.min(), logll.max(), color='k', linestyle='--', label='pred a')
    axes[i].legend()
    peak_i = torch.argmax(logll)
    # higher_than_80quantile = torch.where(logll[center] - torch.quantile(logll, 0.8, dim=-1)>0, True, False)
    axes[i].set_title(f'peak {peak_i}')
    # all_logll[i] = logll
    # print(next_state[show_idx])
  # f.close()
  peak_idx = torch.argmax(all_logll, dim=1)
  peak_score = torch.where(peak_idx==action.reshape(-1), 1., 0.).mean()
  # higher_than_mean = torch.where(all_logll[:, (bins-1)//2] - all_logll.mean(1)>0, 1., 0.).mean()
  # quantile = torch.quantile(all_logll, 0.8, dim=-1)
  # higher_than_80quantile = torch.where(all_logll[:, center] - quantile>0, 1., 0.).mean()
  plt.suptitle(f'sample {sample_idx}, peak score: {peak_score}')
  plt.tight_layout()
  fig_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/profile_ll_action.png'
  if not os.path.exists(os.path.dirname(fig_path)):
    os.makedirs(os.path.dirname(fig_path))
  plt.savefig(fig_path)
  print(fig_path)
  return

def action_profile_likelihood_discrete_batch(args, dataset, agent):
  torch.set_printoptions(threshold=torch.inf)
  batch_size = 256
  times = 10
  metric_matrix = np.zeros((times, 2))
  linear_model_metric_matrix = np.zeros((times, 1))
  linear_model = torch.load('./kms/linear_model_discrete.pth')
  action_dim = agent.action_dim
  n_action_dim = agent.action_dim//agent.n_action
  n_action = agent.n_action

  for i in range(times):
    state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.sample(batch_size))
    assert next_state.shape == (batch_size, agent.state_dim)
    # incremental_matrix = torch.eye(agent.action_dim//agent.n_action).reshape(agent.action_dim//agent.n_action, 1, agent.action_dim//agent.n_action).repeat(1, agent.n_action, 1)
    # assert incremental_matrix.shape == (n_action_dim, n_action, n_action_dim)
    new_action = action.reshape(batch_size, 1, 1, n_action_dim).repeat(1, n_action_dim, n_action, 1)
    for j in range(n_action_dim):
      new_action[:,j,:,j] = torch.arange(n_action)
    # new_action_onehot = torch.eye(n_action)[new_action].reshape(batch_size, n_action_dim, n_action, n_action*n_action_dim)
    state_batch = state.reshape(batch_size, 1, 1, agent.state_dim).repeat(1, n_action_dim, n_action, 1)
    #TODO TODO following
    task_batch = task.reshape(batch_size, 1, 1, 1).repeat(1, n_action_dim, n_action, 1)
    logll = agent.action_loglikelihood(state_batch, new_action, task_batch)[1].detach().cpu()
    print('logll:', logll.shape)
    assert logll.shape == (batch_size, n_action_dim, n_action)
    peak_idx = torch.argmax(logll, dim=-1)
    peak_score = torch.where(peak_idx==action, 1., 0.)
    peak_score = peak_score.mean()
    logll_action = torch.gather(logll,2,action.long().unsqueeze(-1)).squeeze(-1)
    higher_than_mean = torch.where(logll_action - logll.mean(-1)>0, 1., 0.).mean()
    # quantile = torch.quantile(logll, 0.80, dim=-1)
    # higher_than_80quantile = torch.where(logll[..., center] - quantile>0, 1., 0.).mean()
    print(f'peak score: {peak_score}, higher than mean: {higher_than_mean}')
    metric_matrix[i] = peak_score, higher_than_mean
    # linear model
    action_pred = linear_model.predict(state.detach().cpu().numpy())
    action_pred = torch.FloatTensor(action_pred)
    action_pred = torch.clip(action_pred, 0, n_action-1).long()
    assert action_pred.shape == action.shape
    linear_model_logll = agent.action_loglikelihood(state, action_pred, task)[0].detach().cpu()
    # print('linear model logll:', linear_model_logll.shape, 'logll:', logll.shape, 'quantile:', quantile.shape)
    assert linear_model_logll.shape == (batch_size, )
    print('linear model logll:', linear_model_logll)
    print('logll mean:', logll.mean(-1).mean(-1))
    # print('quantile:', quantile.mean(-1))
    higher_than_mean_linear = torch.where(linear_model_logll - logll.mean(-1).mean(-1)>0, 1., 0.).mean()
    # higher_than_80quantile_linear = torch.where(linear_model_logll - quantile.mean(-1)>0, 1., 0.).mean()
    linear_model_metric_matrix[i] = higher_than_mean_linear
  # print('metric:', metric_matrix.mean(0), metric_matrix.std(0)) 
  metric_mean = metric_matrix.mean(0)
  metric_std = metric_matrix.std(0)
  linear_metric_mean = linear_model_metric_matrix.mean(0) 
  linear_metric_std = linear_model_metric_matrix.std(0)
  text_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/action_metric.txt'
  if not os.path.exists(os.path.dirname(text_path)):
    os.makedirs(os.path.dirname(text_path))
  with open(text_path, 'w') as f:
    f.write(f'batch size: {batch_size}, times: {times}\n')
    f.write(f'peak score: {metric_mean[0]:.4f} +- {metric_std[0]:.4f}\n')
    f.write(f'higher than mean: {metric_mean[1]:.4f} +- {metric_std[1]:.4f}\n')
    # f.write(f'higher than 80 quantile: {metric_mean[2]:.4f} +- {metric_std[2]:.4f}\n')
    f.write(f'linear model higher than mean: {linear_metric_mean[0]:.4f} +- {linear_metric_std[0]:.4f}\n')
    # f.write(f'linear model higher than 80 quantile: {linear_metric_mean[1]:.4f} +- {linear_metric_std[1]:.4f}\n')
  print(text_path)
  return

def profile_likelihood(args, dataset, agent):
  scale_factor = args.scale_factor
  torch.set_printoptions(threshold=torch.inf)
  sample_idx = 6
  state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take(sample_idx))
  fig, axes = plt.subplots(4,4, figsize=(10,10))
  axes = axes.flatten()
  bins = 21
  all_logll = torch.zeros(agent.state_dim, bins)
  show_idx = 0
  # linspaces = torch.stack([torch.linspace(next_state[show_idx,i]-1, next_state[show_idx,i]+1, bins) for i in range(agent.state_dim)], 0).transpose(1,0)
  # f = open(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/stateprimeone.txt', 'w')
  # f = open(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/state_one.txt', 'w')
  for i in range(agent.state_dim):
    state_i_scan = torch.linspace(next_state[show_idx,i]-10/scale_factor, next_state[show_idx,i]+10/scale_factor, bins)
    logll = torch.zeros((bins,))
    for j in range(bins):
      new_next_state = next_state.clone() 
      new_next_state[show_idx,i] = state_i_scan[j]
      # new_next_state[show_idx, i+1] = state_i_1_scan[j]
      # print('next_state:', next_state[show_idx])
      # print('new_next_state:', new_next_state[show_idx])/
      # f.write(f'{new_next_state[show_idx]}\n')
      # f.write(f'{state[0]}\n')
      # f.write(f'{action[0]}\n')
      logll[j] = agent.state_likelihood(state, action, new_next_state)[0].detach().cpu()
      # f.write(f'{logll[j]}\n')
      # print(logll[j])
    all_logll[i] = logll
    axes[i].plot(state_i_scan, logll)
    # print(torch.exp(logll))
    # print(logll)
    # expect_next_state = state + action
    expect_next_state = next_state
    axes[i].vlines(expect_next_state[show_idx, i], logll.min(), logll.max(), color='r', linestyle='--', label='next s')
    axes[i].vlines(state[show_idx, i], logll.min(), logll.max(), color='g', linestyle='--', alpha=0.5, label='s')
    axes[i].legend()
    peak_i = torch.argmax(logll)
    axes[i].set_title(f'peak {peak_i}')
    # all_logll[i] = logll
    # print(next_state[show_idx])
  # f.close()
  center = (bins-1)//2
  peak_idx = torch.argmax(all_logll, dim=1)
  peak_score = torch.where(peak_idx==center, 1., 0.).mean()
  higher_than_mean = torch.where(all_logll[:, (bins-1)//2] - all_logll.mean(1)>0, 1., 0.).mean()
  stay_likelihood = agent.state_likelihood(state, action, state)[0]
  higher_than_stay = torch.where(all_logll[:, (bins-1)//2] - stay_likelihood>=-1e-6, 1., 0.).mean()
  print(all_logll[:, (bins-1)//2],stay_likelihood)
  print(all_logll[:, (bins-1)//2] - stay_likelihood)
  # print('stay_likelihood:', stay_likelihood, 'sprime likelihood:',all_logll[:, (bins-1)//2])
  # print('higher_than_stay:', all_logll[:, (bins-1)//2] - stay_likelihood)
  # print('peak score:', peak_score, 'higher than mean:', higher_than_mean, 'higher than stay:', higher_than_stay)
  plt.suptitle(f'sample {sample_idx}, peak score: {peak_score}, higher than mean: {higher_than_mean}, than stay: {higher_than_stay}')
  plt.tight_layout()
  fig_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/profile_ll.png'
  if not os.path.exists(os.path.dirname(fig_path)):
    os.makedirs(os.path.dirname(fig_path))
  plt.savefig(fig_path)
  print(fig_path)
  # profile_likelihood_batch(args, dataset, agent)
  return

def profile_likelihood_batch(args, dataset, agent):
  torch.set_printoptions(threshold=torch.inf)
  scale_factor = args.scale_factor
  batch_size = 256
  times = 10
  bins = 21
  metric_matrix = np.zeros((times, 4))
  state_dim = agent.state_dim
  action_dim = agent.action_dim
  for i in range(times):
    # sample_idxs = np.random.randint(0, dataset.size, batch_size)
    # print('sample_idxs:', sample_idxs)
    state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.sample(batch_size))
    # print('state:', state, 'action:', action, 'next_state:', next_state)
    # assert torch.isclose(state + action, next_state, atol=1e-6).all()
    assert next_state.shape == (batch_size, agent.state_dim)
    center = (bins-1)//2
    total_range = 20/scale_factor
    incremental_matrix = torch.eye(agent.state_dim).reshape(agent.state_dim, 1, agent.state_dim).repeat(1, bins, 1) \
                      * ((torch.arange(bins) - center) * total_range/(bins-1)).reshape(1, bins, 1)
    assert incremental_matrix.shape == (agent.state_dim, bins, agent.state_dim)
    new_next_state = next_state.reshape(batch_size, 1, 1, agent.state_dim) + incremental_matrix.reshape(1, agent.state_dim, bins, agent.state_dim)
    state_batch = state.reshape(batch_size, 1, 1, agent.state_dim).repeat(1, agent.state_dim, bins, 1)
    action_batch = action.reshape(batch_size, 1, 1, agent.action_dim).repeat(1, agent.state_dim, bins, 1)
    # with open(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/state.txt', 'w') as f:
    #   for k in range(agent.state_dim):
    #     for l in range(bins):
    #       f.write(f'{state_batch[0,k,l]}\n')
    #       f.write(f'{action_batch[0,k,l]}\n')
    # for k in range(agent.state_dim):
    #   for l in range(bins):
    #     print('new_next_state:', new_next_state[0,l])
    # print(state_batch.dtype, action_batch.dtype, new_next_state.dtype)
    logll = agent.state_likelihood(state_batch.reshape(-1,state_dim), action_batch.reshape(-1,action_dim), 
                                   new_next_state.reshape(-1,state_dim))[0].reshape(batch_size, agent.state_dim, bins)
    assert logll.shape == (batch_size, agent.state_dim, bins)
    # with open(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/logll.txt', 'w') as f:
    #   for k in range(agent.state_dim):
    #     for l in range(bins):
    #       f.write(f'{logll[0,k,l]}\n')
    peak_idx = torch.argmax(logll, dim=-1)
    # print('peak_idx:', peak_idx.shape)
    # print('peak_idx:', peak_idx, 'logll:', logll[...,peak_idx], 'center:', logll[...,center])
    peak_score = torch.where(peak_idx==center, 1., 0.)
    # print('peak_score:', peak_score.shape)
    # print(np.where(peak_score<1), peak_score[peak_score<1])
    peak_score = peak_score.mean()
    higher_than_mean = torch.where(logll[..., center] - logll.mean(-1)>0, 1., 0.).mean()
    stay_likelihood = agent.state_likelihood(state, action, state)[0]
    higher_than_stay = torch.where(logll[..., center] - stay_likelihood.view(batch_size,1)>=-1e-6, 1., 0.).mean()
    quantile = torch.quantile(logll, 0.90, dim=-1)
    higher_than_90quantile = torch.where(logll[..., center] - quantile>0, 1., 0.).mean()
    print(f'peak score: {peak_score}, higher than mean: {higher_than_mean}, than stay: {higher_than_stay}, higher than 90 quantile: {higher_than_90quantile}')
    metric_matrix[i] = peak_score, higher_than_mean, higher_than_stay, higher_than_90quantile
  print('metric:', metric_matrix.mean(0), metric_matrix.std(0)) 
  metric_mean = metric_matrix.mean(0)
  metric_std = metric_matrix.std(0)
  text_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/metric.txt'
  if not os.path.exists(os.path.dirname(text_path)):
    os.makedirs(os.path.dirname(text_path))
  with open(text_path, 'w') as f:
    f.write(f'total range: {total_range}, bins: {bins}, batch size: {batch_size}, times: {times}\n')
    f.write(f'peak score: {metric_mean[0]:.4f} +- {metric_std[0]:.4f}\n')
    f.write(f'higher than mean: {metric_mean[1]:.4f} +- {metric_std[1]:.4f}\n')
    f.write(f'higher than stay: {metric_mean[2]:.4f} +- {metric_std[2]:.4f}\n')
    f.write(f'higher than 90 quantile: {metric_mean[3]:.4f} +- {metric_std[3]:.4f}\n')
  print(text_path)
  return

def show_phi_weight(args, dataset, agent):
  phi_weight = agent.phi.l1.weight.detach().cpu().numpy()
  save_dir_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)
  n_input = phi_weight.shape[1]
  n_s = n_input//2
  hidden_dim = phi_weight.shape[0]
  def PCA_weight(weight, n_components, save_dir_path, feature_dim, n_s):
    print('weight:', weight)
    assert weight.shape == (feature_dim, n_s*2)
    pca = PCA(n_components=n_components)
    pca.fit(weight.T)
    transformed_weight = pca.transform(weight.T)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.plot(pca.explained_variance_ratio_.cumsum(), marker='o', markersize=5)
    ax.set_title('explained variance ratio')
    plt.savefig(f'{save_dir_path}/phi_weight_evr.png')
    plt.close()
    print(f'{save_dir_path}/phi_weight_evr.png')
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.imshow(transformed_weight, interpolation='nearest')
    plt.savefig(f'{save_dir_path}/phi_weight_pca.png')
    plt.close()
    print(f'{save_dir_path}/phi_weight_pca.png')
    edges, state_name, n_dim = get_edges(n_s)
    transformed_weight_xy = transformed_weight.T.reshape(n_components, n_s, 2)
    # print('transformed_weight:', transformed_weight_xy)
    plot_IG_skeleton(transformed_weight_xy[...,0]*50, state_name, n_components, f'{save_dir_path}/phi_weight_skeleton_x.png')
    plot_IG_skeleton(transformed_weight_xy[...,1]*50, state_name, n_components, f'{save_dir_path}/phi_weight_skeleton_y.png')
    return pca
  def plot_weight_skeleton(weight, save_dir_path, feature_dim, n_s):
    edges, state_name, n_dim = get_edges(n_s)
    assert weight.shape == (feature_dim, n_s*2)
    weight_xy = weight.reshape(feature_dim, n_s, 2)
    plot_IG_skeleton(weight_xy[...,0]*50, state_name, feature_dim, f'{save_dir_path}/phi_weight_skeleton_x.png')
    plot_IG_skeleton(weight_xy[...,1]*50, state_name, feature_dim, f'{save_dir_path}/phi_weight_skeleton_y.png')
    return
  def plot_weight_matrix(weight, save_dir_path, feature_dim, n_s):
    assert weight.shape == (feature_dim, n_s*2)
    state_action_corr = np.sum(phi_weight[:,:n_s]*phi_weight[:,n_s:], axis=1)/np.sqrt(np.sum(phi_weight[:,:n_s]**2, axis=1)*np.sum(phi_weight[:,n_s:]**2, axis=1))
    corr = np.correlate(phi_weight[:,:n_s].flatten(), phi_weight[:,n_s:].flatten())
    print('corr:', corr)
    corr_value = state_action_corr.mean()
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    img = ax.imshow(weight.T, cmap='hot', interpolation='nearest')
    fig.colorbar(img, ax=ax)
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.suptitle(f'state action correlation:{corr_value}')
    save_path = f'{save_dir_path}/phi_weight_matrix.png'
    plt.savefig(save_path)
    print(save_path)
    plt.close()
    return
  plot_weight_matrix(phi_weight, save_dir_path, hidden_dim, n_s)
  # few_hots = np.where(np.abs(phi_weight) > np.mean(np.abs(phi_weight), axis=1, keepdims=True), 1, 0).mean(1)
  # fig, ax = plt.subplots(1,1,figsize=(5,5))
  # ax.hist(few_hots, bins=20, density=True, alpha=0.6, color='orange')
  # U, S, Vh = np.linalg.svd(phi_weight, full_matrices=False)
  # print('Ushape:', U.shape, 'Sshape:', S.shape, 'Vshape:', Vh.shape)
  # print('singular value:', S)
  # assert np.allclose(phi_weight, U[...,:S.shape[0]]@np.diag(S)@Vh, atol=1e-6)

  # PCA_weight(phi_weight, 20, save_dir_path, hidden_dim, n_s)
  # plot_weight_skeleton(phi_weight, save_dir_path, hidden_dim, n_s)

  # fig, axes = plt.subplots(1,2, figsize=(10,5))
  # for i in range(hidden_dim):
    # if i % 17 == 0:
      # axes[0].plot(phi_weight[i,:n_s], label=i)
      # axes[1].plot(phi_weight[i, n_s:], label=i)
  # state_action_corr = np.sum(phi_weight[:,:n_s]*phi_weight[:,n_s:], axis=1)/np.linalg.norm(phi_weight[:,:n_s],axis=1)*np.linalg.norm(phi_weight[:,n_s:], axis=1)
  # state_action_corr = np.corrcoef(phi_weight[:,:n_s], phi_weight[:,n_s:], rowvar=True)
  # plt.suptitle(f'correlation between state and action, mean:{state_action_corr.mean()}')
  # print('state action correlation:', state_action_corr)
  # axes[0].legend()
  # axes[0].set_title('state')
  # axes[1].legend()
  # axes[1].set_title('action')

  # fig, axes = plt.subplots(1,3, figsize=(15,5))

  # ax.spines['top'].set_visible(True)
  # ax.set_xticks(np.arange(hidden_dim), np.arange(hidden_dim), fontsize=30, color='k')
  # ax.xaxis.set_ticks_position('top')
  
  # ax.plot(S, c='r', markersize=10, linewidth=2, marker='o')
  # fig, axes = plt.subplots(2,2, figsize=(10,10))
  # axes = axes.flatten()
  # for i in range(4):
  #   for j in range(n_input//2):
  #     if j % 7 == 0:
  #       axes[i].plot(Vh[i//2*n_s+j, i%2*n_s:(i%2+1)*n_s])
  # corr_ar = np.sum(Vh[:,:n_s]*Vh[:,n_s:], axis=1)/np.linalg.norm(Vh[:,:n_s],axis=1)*np.linalg.norm(Vh[:,n_s:], axis=1)
  # print('pos correlation:', corr_ar[:n_s])
  # print('neg correlation:', corr_ar[n_s:])
    
  # ax.imshow(Vh)
  # n_input = phi_weight.shape[1]
  # ax.spines['top'].set_visible(True)
  # ax.set_xticks(np.arange(n_input//2), np.arange(n_input//2), fontsize=30, color='k')
  # ax.xaxis.set_ticks_position('top')
  # ax.set_xticks(np.arange(n_input//2, n_input), np.arange(n_input//2), fontsize=30, color='red')
  # ax.set_yticks(np.arange(n_input), np.arange(n_input), fontsize=30)
  # print('U:', U)
  # print('V:', V)
  # ax.imshow(phi_weight.T,)
  # axes[0].imshow(phi_weight, cmap='hot', interpolation='nearest')
  # axes[1].hist(phi_weight.flatten(), bins=20, density=True, color='orange')
  # for i in range(phi_weight.shape[0]):
  #   axes[2].plot(phi_weight[i], label=i)
  # plt.legend()
  # plt.suptitle('phi weight')

  # plt.savefig(save_path)
  # print(save_path)
  # plt.close()
  return
  
def show_uw(args, dataset, agent):
  print('n_task:', agent.n_task)
  task_all = torch.eye(agent.n_task)
  # u1, u2 = agent.critic(task_all)
  u_all = agent.u(task_all)
  w_all = agent.w(task_all)
  for name, param in agent.u.named_parameters():
    print(name, param)
  params = dict(agent.u.named_parameters())
  weight = params['trunk.0.weight'].detach().cpu().numpy()
  print('weight:', weight.shape)
  # bias = params['trunk.0.bias'].detach().cpu().numpy()  
  # print('bias:', bias.shape)
  # u = u1.detach().cpu().numpy()
  u = u_all.detach().cpu().numpy()
  w = w_all.detach().cpu().numpy()
  # print(u)
  save_dir_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)
  fig, axes = plt.subplots(3,1, figsize=(15,15))
  # fig.colorbar(axes.imshow(u, cmap='hot', interpolation='nearest'))
  for i in [2,3]:
    # predicted_ui = weight[:,i] + bias
    # axes[0].plot(predicted_ui, label=i)
    axes[0].plot(u[i], label=i)
    large_idx = np.argsort(np.abs(u[i]))[::-1]
    print('large idx:', large_idx[:5])
    print('large value:', u[i][large_idx[:5]])
    print(i, 'u:', u[i, 74])
    axes[1].plot(weight[:,i], label=i)
    # axes[1].plot(w[i], label=i)
    # print('u:', u[i], 'w:', w[i])
  # axes[2].plot(bias, label='bias')
  axes[0].set_title('u')
  # axes[1].set_title('w')
  axes[0].legend()
  axes[1].legend()
  axes[2].legend()
  # axes[1].hist(u.flatten(), bins=20, density=True, color='orange')
  save_path = f'{save_dir_path}/uw.png'
  plt.savefig(save_path)
  print(save_path)
  plt.close()

  return

def show_last_weight(args, dataset, agent):
  for name, param in agent.critic.named_parameters():
    print(name, param.shape)
  last_layer_weight = agent.critic.named_parameters()['trunk.2.weight'][0].detach().cpu().numpy()



def density_trajectory(args, dataset, agent):
  scale_factor = args.scale_factor
  task = np.arange(0, agent.n_task)
  timestep = 10
  for t in range(agent.n_task):
    state_seq = np.zeros((timestep, agent.state_dim))
    # sample_task = -1
    idx = np.where(dataset.task == t)[0][0]
    print('task:', t, 'idx:', idx)
    state, action, next_state, reward, done, task, next_task = unpack_batch(dataset.take([idx]))
    print('state:', state.shape, 'action:', action.shape, 'next_state:', next_state.shape, 'task:', task.shape)
    # print(state.shape)
    state_seq[0] = state[0].detach().cpu().numpy()
    for i in range(timestep):
      optimized_action = optimize_action(scale_factor, action, state, task, agent)
      assert optimized_action.shape == (1, agent.action_dim)
      next_state = state + optimized_action
      state_seq[i] = next_state[0].detach().cpu().numpy()
      state = next_state.clone().detach()
      task = task.clone().detach()
      action = optimized_action.clone().detach()  
      # action = optimized_action
    plot_gif(state_seq, f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/density_rollout_{t}.gif')  
    
  return

def optimize_action(scale_factor, action, state, task, agent):
  # feature_dim = agent.feature_dim
  # state_dim, action_dim = agent.state_dim, agent.action_dim
  # batch_size = 1
  # batch = dataset.sample(batch_size)
  # state, action, next_state, reward, done, task, next_task = unpack_batch(batch)
  optimized_action_logits = torch.randn_like(action)/scale_factor
  optimized_action_logits.requires_grad = True
  optimizer = torch.optim.Adam([optimized_action_logits], lr=0.1)
  iteration_time = 1000
  loss_list = []
  for i in range(iteration_time):
    # print('loss:', loss.item())
    # print(i)
    # optimized_action_logits = optimized_action_logits.clone().detach().requires_grad_()
    optimized_action = torch.tanh(optimized_action_logits)/20
    action_logll = agent.action_loglikelihood(state, optimized_action, task)[0]
    loss = -action_logll.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # loss_list.append(loss.item())
    # print('action:', action, 'optimized_action:', optimized_action)
  # plt.plot(loss_list)
  # plt.show()
  return optimized_action

EPS_GREEDY = 0.01

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default=0, type=str)                     
  parser.add_argument("--alg", default="diffsrsac")                     # Alg name (sac, vlsac, spedersac, ctrlsac, mulvdrq, diffsrsac, spedersac)
  parser.add_argument("--env", default="HalfCheetah-v4")          # Environment name
  parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--start_timesteps", default=25e3, type=float)# Time steps initial random policy is used
  parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
  parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
  parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
  parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
  parser.add_argument("--hidden_dim", default=256, type=int)      # Network hidden dims
  parser.add_argument("--feature_dim", default=256, type=int)      # Latent feature dim
  parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
  parser.add_argument("--tau", default=0.005)                     # Target network update rate
  parser.add_argument("--learn_bonus", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--extra_feature_steps", default=3, type=int)
  parser.add_argument("--ckpt_n", default=0, type=int)
  parser.add_argument("--times", default=3, type=int)
  parser.add_argument("--device", default='cuda:0', type=str)
  parser.add_argument("--scale_factor", default=1, type=float)
  args = parser.parse_args()


  replay_buffer, state_dim, action_dim, n_task = load_keymoseq('test', args.dir, args.device)
  save_path = f'model/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    # "action_space": gym.spaces.Box(-1, 1, (action_dim,), dtype=np.float32),
    "discount": args.discount,
    # "tau": args.tau,
    # "hidden_dim": args.hidden_dim,
  }

  kwargs['extra_feature_steps'] = 2
  kwargs['phi_and_mu_lr'] = 0.0001
  kwargs['phi_hidden_dim'] = 512
  kwargs['phi_hidden_depth'] = 1
  kwargs['mu_hidden_dim'] = 512
  kwargs['mu_hidden_depth'] = 1
  kwargs['critic_and_actor_lr'] = 0.0003
  kwargs['critic_and_actor_hidden_dim'] = 256
  kwargs['feature_dim'] = args.feature_dim
  kwargs['device'] = args.device
  # kwargs['state_task_dataset'] = replay_buffer.state
  kwargs['learnable_temperature'] = False
  kwargs['n_task'] = n_task
  kwargs['tau'] = args.tau
  kwargs['hidden_dim'] = args.hidden_dim  
  kwargs['directory'] = args.dir
  agent = spedersac_agent.SPEDERSACAgent(**kwargs)
  # agent = spedersac_agent.QR_IRLAgent(**kwargs)
  # agent = spedersac_agent.SimpleWorldModel(**kwargs)
  # agent = spedersac_agent.RandomFeatureModel(**kwargs)
  
  # agent.load_phi_mu(torch.load(f'{save_path}/checkpoint_{args.max_timesteps}.pth'))
  agent.load_state_dict(torch.load(f'{save_path}/checkpoint_{args.max_timesteps}.pth'))
  # agent.load_actor(torch.load(f'{save_path}/checkpoint_{args.max_timesteps}.pth'))
  # print('load model from:', f'{save_path}/checkpoint_{args.max_timesteps}.pth')

  # show_uw(args, replay_buffer, agent)
  rollout_check_profile_all(args, replay_buffer, agent)
  # profile_likelihood(args, replay_buffer, agent)
  # profile_likelihood_batch(args, replay_buffer, agent)
  # action_profile_likelihood(args, replay_buffer, agent)
  # action_profile_likelihood_batch(args, replay_buffer, agent)
  # action_test_logll(args, replay_buffer, agent)
  # test_logll_smoothly(args, replay_buffer, agent)
  # posll, posstd, negll, negstd = test_logll(args, replay_buffer, agent)
  # draw_IG_skeleton(args, replay_buffer, agent)

  # show_phi_weight(args, replay_buffer, agent)
  # show_last_weight(args, replay_buffer, agent)
  # action_profile_likelihood_discrete(args, replay_buffer, agent)
  # action_profile_likelihood_discrete_batch(args, replay_buffer, agent)
  # optimize_action(args, replay_buffer, agent)
  # density_trajectory(args, replay_buffer, agent)
  # optimize_next_state(args, replay_buffer, agent)

  # optimize_input(args, agent)
  # cluster_in_phi_space(args, replay_buffer, agent)
  # args.times = 3
  # IntegratedGradients_attr(args, replay_buffer, agent)
  # get_edges()
  # PCA_IG_skeleton(args, replay_buffer, agent)
  # visualize_wu(args, replay_buffer, agent)
  # action_loglikelihood(args, replay_buffer, agent)
  # action_profile(args, replay_buffer, agent)
  # check_action_space(args, replay_buffer, agent)
  # rollout(args, replay_buffer, agent, 2)
  # rollout_check_profile(args, replay_buffer, agent, 2)

  # rollout_multiple_syllables(args, replay_buffer, agent)
  # action_loglikelihood_multiple_syllables(args, replay_buffer, agent)



