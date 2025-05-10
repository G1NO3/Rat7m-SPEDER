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

def plot_gif_onefig(stateseq, save_path):
  # stateseq: [timestep, state_dim]
  print(stateseq.shape)
  edges, state_name, n_dim = get_edges(stateseq.shape[-1])
  fig, axis = plt.subplots(1, 1, figsize=(5, 5))
  n_bodyparts = len(state_name)
  n_img = stateseq.shape[0]
  state_seqs_to_plot = stateseq.reshape(n_img, n_bodyparts, 2)
  cmap = plt.cm.get_cmap('viridis')
  keypoint_colors = cmap(np.linspace(0, 1, len(state_name)))
  state_seqs_to_plot -= state_seqs_to_plot.mean(axis=(0,1), keepdims=True)
  axmin = -0.2
  axmax = 0.2
  aymin = -0.2
  aymax = 0.2
  ymin = np.min(state_seqs_to_plot[...,1], axis=(-1,-2))
  ymax = np.max(state_seqs_to_plot[...,1], axis=(-1,-2))
  xmin = np.min(state_seqs_to_plot[...,0], axis=(-1,-2))
  xmax = np.max(state_seqs_to_plot[...,0], axis=(-1,-2))
  indicator = np.where((aymin > ymin) | (aymax < ymax) | (axmin > xmin) | (axmax < xmax), 1, 0)
  aymin = np.where(indicator, -0.6, aymin)
  aymax = np.where(indicator, 0.6, aymax)
  axmin = np.where(indicator, -0.6, axmin)
  axmax = np.where(indicator, 0.6, axmax)
  for i in range(n_img):
    if i % 3 == 1 or i % 3 == 2:
      continue
    for p1, p2 in edges:
      axis.plot(
          *state_seqs_to_plot[i, (p1, p2)].T,
          color=keypoint_colors[p1],
          linewidth=5.0,zorder=i*3)
    axis.scatter(
        *state_seqs_to_plot[i].T,
        c=keypoint_colors,
        s=100,zorder=i*3+1)
    # draw a white rectangle
    if i < n_img - 1:
        axis.fill_between([axmin, axmax], y1=aymin, y2=aymax, color='white', alpha=0.1,
                        zorder=i*3+2)
    axis.set_xlim(axmin, axmax)
    axis.set_ylim(aymin, aymax)
  save_fig(save_path)


def pair_gif_and_u(stateseq, u_matrix, taskseq, save_path):
  # stateseq: [timestep, state_dim]
  # u_matrix: [timestep, feature_dim]
  # taskseq: [timestep, 1]
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
  aymin = np.where(indicator, -0.6, aymin)
  aymax = np.where(indicator, 0.7, aymax)
  axmin = np.where(indicator, -0.7, axmin)
  axmax = np.where(indicator, 0.6, axmax) 
  fig = plt.figure(figsize=(16, 4))
  ax0 = fig.add_axes([0.05, 0.12, 0.2, 0.8])
  ax1 = fig.add_axes([0.3, 0.12, 0.68, 0.8])
  ax_kmslabel = fig.add_axes([0.3, 0.05, 0.68, 0.07])
  ax_kmslabel.axis('off')
  ax0.axis('off')
  rasters = []
  for i in range(timestep):
    ax0.clear()
    ax1.clear()
    ax_kmslabel.clear()
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
        s=100,zorder=0)
    for j in range(u_matrix.shape[1]):
      ax1.plot(u_matrix[:,j], label=f'{j}')
    ax1.vlines(i, ymin=u_matrix.min(), ymax=u_matrix.max(), color='black', linestyle='--')
    ax1.set_ylim(u_matrix.min(), u_matrix.max())
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xticks([])
    u_matrix_idx = np.argsort(u_matrix, 1)
    ax1.set_title(f'{i}, first:{u_matrix_idx[i,-1], u_matrix_idx[i,-2], u_matrix_idx[i,-3]}, \
                  last:{u_matrix_idx[i,2], u_matrix_idx[i,1], u_matrix_idx[i,0]}')
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
  convert_gif_to_mp4(save_path)

