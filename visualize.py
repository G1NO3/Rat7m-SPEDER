# import gym.spaces
import numpy as np
import torch
import gym
import argparse
import os, glob

from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.vlsac import vlsac_agent
from agent.ctrlsac import ctrlsac_agent
from agent.diffsrsac import diffsrsac_agent
from agent.spedersac import spedersac_agent
from main import load_rat7m, load_halfcheetah
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

def IntegratedGradients_attr(args, dataset, agent):
  model = agent.phi
  ig = IntegratedGradients(model)
  # sa_ar = torch.zeros((args.times, args.batch_size, agent.state_dim+agent.action_dim))
  ig_matrix_all = torch.zeros((args.times, agent.feature_dim, agent.state_dim+agent.action_dim))
  ig_std_matrix_all = torch.zeros((args.times, agent.feature_dim, agent.state_dim+agent.action_dim))
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
  for i in range(0, args.times):
    batch = dataset.sample(args.batch_size)
    state, action, next_state, reward, done = unpack_batch(batch)
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
  fig, ax = plt.subplots(1,2, figsize=(20, 10))
  # sns.heatmap(ig_matrix, cmap='coolwarm', ax=ax)
  state_name = ['HeadF','HeadB','HeadL','SpineF','SpineM','SpineL','Offset1',\
              'Offset2','HipL','HipR','ElbowL','ArmL','ShoulderL','ShoulderR',
              'ElbowR','ArmR','KneeR','KneeL','ShinL','ShinR']
  action_name = state_name
  print(agent.state_dim, len(np.arange(0,agent.state_dim,3)))
  ig_matrix = ig_matrix_all.mean(dim=0)
  ig_std_matrix = ig_std_matrix_all.mean(dim=0)
  np.save(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_matrix.npy', ig_matrix)
  np.save(f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ig_std_matrix.npy', ig_std_matrix)
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
    axes[2*i].plot(ig_matrix[i,:agent.state_dim].detach().cpu().numpy())
    axes[2*i].set_xticks(np.arange(0,agent.state_dim,3), state_name, rotation=45)
    axes[2*i+1].plot(ig_matrix[i,agent.state_dim:].detach().cpu().numpy())
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

def test_logll(args, dataset, agent):
  positive_ll = torch.zeros(args.times)
  negative_ll = torch.zeros(args.times)
  for i in range(0, args.times):
    batch = dataset.sample(args.batch_size)
    batch_2 = dataset.sample(args.batch_size)
    state, action, next_state, reward, done = unpack_batch(batch)
    # print(state.shape, action.shape, next_state.shape)
    s_random, a_random, ns_random, _, _ = unpack_batch(batch_2)
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
  save_path = f'figure/{args.env}/{args.alg}/{args.dir}/{args.seed}/ll.png'
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  plt.savefig(save_path)
  print(save_path)
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
  args = parser.parse_args()


  replay_buffer, state_dim, action_dim = load_rat7m()
  save_path = f'model/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "action_space": gym.spaces.Box(-1, 1, (action_dim,), dtype=np.float32),
    "discount": args.discount,
    "tau": args.tau,
    "hidden_dim": args.hidden_dim,
  }

  kwargs['extra_feature_steps'] = 5
  kwargs['phi_and_mu_lr'] = 0.0001
  kwargs['phi_hidden_dim'] = 512
  kwargs['phi_hidden_depth'] = 1
  kwargs['mu_hidden_dim'] = 512
  kwargs['mu_hidden_depth'] = 0
  kwargs['critic_and_actor_lr'] = 0.0003
  kwargs['critic_and_actor_hidden_dim'] = 256
  kwargs['feature_dim'] = args.feature_dim
  kwargs['device'] = 'cuda:0'
  kwargs['state_task_dataset'] = replay_buffer.state
  agent = spedersac_agent.SPEDERSACAgent(**kwargs)
  
  agent.load_state_dict(torch.load(f'{save_path}/checkpoint_{args.max_timesteps}.pth'))

  args.times = 100
  posll, posstd, negll, negstd = test_logll(args, replay_buffer, agent)
  print('positive likelihood:', posll, posstd)
  print('negative likelihood:', negll, negstd)
  # optimize_input(args, agent)
  # cluster_in_phi_space(args, replay_buffer, agent)
  args.times = 3
  IntegratedGradients_attr(args, replay_buffer, agent)



