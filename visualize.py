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
from main import load_rat7m, load_halfcheetah, load_keymoseq
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

def rollout(args, dataset, agent):
  syllable = 1
  timestep = 3
  while True:
    sample = dataset.sample(args.batch_size)
    task = sample.task
    print(task)
    all_idx = torch.where(task == syllable)[0]
    if len(all_idx) > 0:
      break
  idx = all_idx[0]
  state = sample.state[idx]
  action = sample.action[idx]
  stateseq = torch.zeros((timestep, *state.shape))
  actionseq = torch.zeros((timestep, *action.shape))
  stateseq[0] = state
  actionseq[0] = action
  for i in range(1, timestep):
    state, action, sp_likelihood, ap_q = agent.step(state, syllable, action)
    print(sp_likelihood, ap_q)
    stateseq[i] = state
    actionseq[i] = action
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/rollout.gif'
  plot_gif(stateseq, save_path)
  
def plot_gif(stateseq, save_path):
  # stateseq: [timestep, state_dim]
  edges, state_name = get_edges(stateseq.shape[1])
  fig, axis = plt.subplots(1, 1, figsize=(5, 6))
  n_bodyparts = len(state_name)
  n_img = stateseq.shape[0]
  if stateseq.shape[1] == 54:
    dims, name = [0,2], 'xz'
    state_seq_to_plot = stateseq.reshape(n_img, n_bodyparts, 3)[..., dims]
  else:
    dims, name = [0,1], 'xy'
    state_seq_to_plot = stateseq.reshape(n_img, n_bodyparts, 2)
  cmap = plt.cm.get_cmap('viridis')
  keypoint_colors = cmap(np.linspace(0, 1, len(state_name)))
  rasters = []
  ymin = state_seq_to_plot[:,:,1].min()
  ymax = state_seq_to_plot[:,:,1].max()
  xmin = state_seq_to_plot[:,:,0].min()
  xmax = state_seq_to_plot[:,:,0].max()
  for i in range(n_img):
    axis.clear()
    for p1, p2 in edges:
      axis.plot(
          *state_seq_to_plot[i, (p1, p2)].T,
          color=keypoint_colors[p1],
          linewidth=5.0)
    axis.scatter(
        *state_seq_to_plot[i].T,
        c=keypoint_colors,
        s=100)
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    
    rasters.append(rasterize_figure(fig))

  pil_images = [Image.fromarray(np.uint8(img)) for img in rasters]
  # Save the PIL Images as an animated GIF
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  pil_images[0].save(
      save_path,
      save_all=True,
      append_images=pil_images[1:],
      duration=500,
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


def action_loglikelihood(args, dataset, agent):
  batch_size = args.batch_size
  times = 100
  positive_logll = np.zeros(times)
  negative_logll = np.zeros(times)
  for i in range(times):
    batch_1 = dataset.sample(batch_size)
    batch_2 = dataset.sample(batch_size)
    positive_logll[i] = agent.action_loglikelihood(batch_1.state, batch_1.action, batch_1.task).detach().cpu().numpy()
    negative_logll[i] = agent.action_loglikelihood(batch_1.state, batch_2.action, batch_1.task).detach().cpu().numpy()
  print('pos:', np.nanmean(positive_logll), np.nanstd(positive_logll))
  print('neg:', np.nanmean(negative_logll), np.nanstd(negative_logll))
  t, p = ttest_ind(positive_logll, negative_logll)
  print('t:', t, 'p:', p)
  fig, ax = plt.subplots(1,1, figsize=(10,10))
  ax.hist(positive_logll, bins=20, alpha=0.6, density=True, color='orange')
  ax.hist(negative_logll, bins=20, alpha=0.6, density=True, color='g')
  plt.legend(['positive sample', 'negative sample'])
  plt.title('action log likelihood, p={:.4f}'.format(p))
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
  elif state_dim == 18:
    state_name = ['tail', 'spine4', 'spine3', 'spine2', 'spine1', 'head', 'nose', 'right ear', 'left ear']
    skeleton = [('tail', 'spine4'), ('spine4', 'spine3'), ('spine3', 'spine2'),
                ('spine2', 'spine1'), ('spine1', 'head'), ('head', 'nose'),
                ('head', 'left ear'), ('head', 'right ear')]
  edges = []
  for i in skeleton:
    edges.append((state_name.index(i[0]), state_name.index(i[1])))
  return edges, state_name

def PCA_IG_skeleton(args, dataset, agent):
  # ig_matrix: [feature_dim, state_dim+action_dim]
  ig_matrix, ig_std_matrix, ig_matrix_agg_xyz, ig_std_matrix_agg_xyz, \
    state_name, action_name = cal_IG_matrix(args, dataset, agent, 3)
  print('IG done')
  n_bodyparts = len(state_name)
  pca = PCA(n_components=20)
  assert ig_matrix_agg_xyz.shape == (agent.feature_dim, n_bodyparts*2)
  print(ig_matrix_agg_xyz.shape)  
  ig_pca = pca.fit_transform(ig_matrix_agg_xyz.T)
  # ig_pca = pca.components_
  print(ig_pca.shape)
  assert ig_pca.shape == (n_bodyparts*2, 20)
  fig, ax = plt.subplots(1,1, figsize=(10, 10))
  ax.plot(pca.explained_variance_ratio_.cumsum())
  ax.set_title('PCA explained variance ratio')
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/pca_evr.png'
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  print(save_path)
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_pca.png'
  plot_IG_skeleton(ig_pca.T, state_name, 20, save_path)

def draw_IG_skeleton(args, dataset, agent):
  ig_matrix, ig_std_matrix, ig_matrix_agg_xyz, ig_std_matrix_agg_xyz, \
    state_name, action_name = cal_IG_matrix(args, dataset, agent, 3)
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_skeleton.png'
  plot_IG_skeleton(ig_matrix_agg_xyz, state_name, agent.feature_dim, save_path)

def plot_IG_skeleton(ig_matrix_agg_xyz, state_name, feature_dim, save_path):
  # ig_matrix: [feature_dim, state_dim+action_dim]
  edges, state_name = get_edges(ig_matrix_agg_xyz.shape[1]//2)
  col = 10*2
  row = feature_dim//(col//2) + 1
  fig, axes = plt.subplots(row, col, figsize=(col*5, row*6))
  axes = axes.flatten()
  ymean = np.load('./ymean.npy')
  dims, name = [0,2], 'xz'
  cmap = plt.cm.get_cmap('viridis')
  keypoint_colors = cmap(np.linspace(0, 1, len(state_name)))
  n_bodyparts = len(state_name)
  assert ymean.shape == (n_bodyparts, 3)
  assert ig_matrix_agg_xyz.shape == (feature_dim, 2*n_bodyparts)
  for i in range(feature_dim):
    for e in edges:
      axes[i*2].plot(
          *ymean[:, dims][e,:].T,
          color=keypoint_colors[e[0]],
          linewidth=5.0,
          zorder=0)
      axes[i*2+1].plot(
          *ymean[:, dims][e,:].T,
          color=keypoint_colors[e[0]],
          linewidth=5.0,
          zorder=0)
    node_colors = ['blue' if ig_matrix_agg_xyz[i, j] < 0 else 'red' for j in range(2*n_bodyparts)]
    axes[i*2].scatter(
          *ymean[:, dims].T,
          c=node_colors[:n_bodyparts],
          s=np.abs(ig_matrix_agg_xyz[i, :n_bodyparts])*120,
          zorder=1)
    axes[i*2+1].scatter(
          *ymean[:, dims].T,
          c=node_colors[n_bodyparts:],
          s=np.abs(ig_matrix_agg_xyz[i, n_bodyparts:])*120,
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
  state_name = ['HeadF','HeadB','HeadL','SpineF','SpineM',
                'SpineL','HipL','HipR','ElbowL','ArmL',
                'ShoulderL','ShoulderR','ElbowR','ArmR','KneeR',
                'KneeL','ShinL','ShinR']
  action_name = state_name
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  if os.path.exists(f'{save_path}/ig_matrix.npy'):
    ig_matrix = np.load(f'{save_path}/ig_matrix.npy')
    ig_std_matrix = np.load(f'{save_path}/ig_std_matrix.npy')
    ig_matrix_agg_xyz = np.load(f'{save_path}/ig_matrix_agg.npy')
    ig_std_matrix_agg_xyz = np.load(f'{save_path}/ig_std_matrix_agg.npy')
    return ig_matrix, ig_std_matrix, ig_matrix_agg_xyz, ig_std_matrix_agg_xyz, state_name, action_name

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
    sa_ar = torch.cat([state, action], dim=1).to('cuda:0')
    sa_ar.requires_grad = True
    for j in range(agent.feature_dim):
      attr_ig, delta = ig.attribute(sa_ar, target=j, return_convergence_delta=True, 
                      baselines=-1)
      ig_matrix_all[i][j] = attr_ig.mean(dim=0).detach().cpu()
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
  print(agent.state_dim, len(np.arange(0,agent.state_dim,3)))
  ig_matrix = ig_matrix_all.mean(dim=0)
  ig_std_matrix = ig_std_matrix_all.mean(dim=0)
  np.save(f'{save_path}/ig_matrix.npy', ig_matrix)
  np.save(f'{save_path}/ig_std_matrix.npy', ig_std_matrix)
  ig_matrix_agg_xyz = ig_matrix.reshape(agent.feature_dim, -1, 3).mean(dim=-1)
  ig_std_matrix_agg_xyz = ig_std_matrix.reshape(agent.feature_dim, -1, 3).mean(dim=-1)
  ig_matrix = ig_matrix.detach().cpu().numpy()
  ig_std_matrix = ig_std_matrix.detach().cpu().numpy()
  ig_matrix_agg_xyz = ig_matrix_agg_xyz.detach().cpu().numpy()
  ig_std_matrix_agg_xyz = ig_std_matrix_agg_xyz.detach().cpu().numpy()
  np.save(f'{save_path}/ig_matrix_agg.npy', ig_matrix_agg_xyz)
  np.save(f'{save_path}/ig_std_matrix_agg.npy', ig_std_matrix_agg_xyz)
  return ig_matrix, ig_std_matrix, ig_matrix_agg_xyz, ig_std_matrix_agg_xyz, state_name, action_name

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

def optimize_input(args, agent):
  feature_dim = agent.feature_dim
  state_dim, action_dim = agent.state_dim, agent.action_dim
  batch_size = 16 
  focus_feature = 0
  # Define the target output
  target_output = torch.zeros((batch_size, feature_dim))
  target_output.requires_grad = False
  target_output[:, focus_feature] = 1.0  # Set the target output for the desired feature  

  # Initialize the input with random values (ensure it has the right shape)
  input_shape = (batch_size, state_dim+action_dim)  # Replace with your model's input shape
  optimized_input = torch.randn(input_shape, requires_grad=True)

  # Define the optimizer to update the input
  optimizer = optim.Adam([optimized_input], lr=0.01)

  # Define the loss function (e.g., MSE)
  loss_fn = nn.MSELoss()

  # Optimization loop
  num_iterations = 1000  # Adjust based on convergence
  model = agent.phi
  for i in range(num_iterations):
      optimizer.zero_grad()

      # Forward pass
      output = model(optimized_input)

      # Compute the loss
      loss = loss_fn(output, target_output)

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
  print("Final optimized input:", final_input)

def test_logll(args, dataset, agent, times=100):
  positive_ll = torch.zeros(times)
  negative_ll = torch.zeros(times)
  for i in range(0, times):
    batch = dataset.sample(args.batch_size)
    batch_2 = dataset.sample(args.batch_size)
    state, action, next_state, reward, done, task, next_task = unpack_batch(batch)
    # print(state.shape, action.shape, next_state.shape)
    s_random, a_random, ns_random, _, _, _, _ = unpack_batch(batch_2)
    positive_ll[i] = agent.log_likelihood(state, action, next_state)
    negative_ll[i] = agent.log_likelihood(state, action, ns_random)
  # print(positive_ll)
  # print(negative_ll)
  # significance test
  t_stat, p_value = ttest_ind(positive_ll, negative_ll, equal_var=False)
  fig, ax = plt.subplots()
  ax.hist(positive_ll.cpu().numpy(), bins=20, density=True, alpha=0.6, color='orange')
  ax.hist(negative_ll.cpu().numpy(), bins=20, density=True, alpha=0.6, color='g')
  plt.legend(['positive sample', 'negative sample'])
  plt.title('likelihood, p={:.4f}'.format(p_value))
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  plt.savefig(f'{save_path}/ll.png')
  print(f'{save_path}/ll.png')
  return positive_ll.mean(), positive_ll.std(), negative_ll.mean(), negative_ll.std()



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
  parser.add_argument("--batch_size", default=64, type=int)      # Batch size for both actor and critic
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
  args = parser.parse_args()


  replay_buffer, state_dim, action_dim, n_task = load_keymoseq('test', args.device)
  save_path = f'model/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  # set seeds
  torch.manual_seed(args.seed+2)
  np.random.seed(args.seed+2)

  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "action_space": gym.spaces.Box(-1, 1, (action_dim,), dtype=np.float32),
    "discount": args.discount,
    "tau": args.tau,
    "hidden_dim": args.hidden_dim,
  }

  kwargs['extra_feature_steps'] = 2
  kwargs['phi_and_mu_lr'] = 0.0001
  kwargs['phi_hidden_dim'] = 512
  kwargs['phi_hidden_depth'] = 1
  kwargs['mu_hidden_dim'] = 512
  kwargs['mu_hidden_depth'] = 0
  kwargs['critic_and_actor_lr'] = 0.0001
  kwargs['critic_and_actor_hidden_dim'] = 256
  kwargs['feature_dim'] = args.feature_dim
  kwargs['device'] = args.device
  kwargs['state_task_dataset'] = replay_buffer.state
  kwargs['learnable_temperature'] = True
  kwargs['n_task'] = n_task
  agent = spedersac_agent.SPEDERSACAgent(**kwargs)
  
  agent.load_state_dict(torch.load(f'{save_path}/checkpoint_{args.max_timesteps}.pth'))
  print('load model from:', f'{save_path}/checkpoint_{args.max_timesteps}.pth')
  # args.times = 100
  # posll, posstd, negll, negstd = test_logll(args, replay_buffer, agent)
  # print('positive likelihood:', posll, posstd)
  # print('negative likelihood:', negll, negstd)
  # optimize_input(args, agent)
  # cluster_in_phi_space(args, replay_buffer, agent)
  # args.times = 3
  # IntegratedGradients_attr(args, replay_buffer, agent)
  # get_edges()
  # draw_IG_skeleton(args, replay_buffer, agent)
  # PCA_IG_skeleton(args, replay_buffer, agent)
  # visualize_wu(args, replay_buffer, agent)
  # action_loglikelihood(args, replay_buffer, agent)
  # check_action_space(args, replay_buffer, agent)
  rollout(args, replay_buffer, agent)


