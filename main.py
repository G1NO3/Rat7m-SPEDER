import gym.spaces
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
from utils.util import unpack_batch
def load_keymoseq(category, directory, device='cuda:0'):
  state_dim = 16
  action_dim = 16
  n_task = 10
  replay_buffer = buffer.ReplayBuffer(state_dim, action_dim, 1000000, device)
  if 'discrete' in directory:
    state_dim = 16
    action_dim = 80
    n_task = 10
    replay_buffer_path = f'./kms/{category}_data_discrete.pth'
  elif '24' in directory:
    replay_buffer_path = f'./kms/{category}_data_24.pth'
  elif '2_only' in directory:
    replay_buffer_path = f'./kms/{category}_data_2_only.pth'
  else:
    replay_buffer_path = f'./kms/{category}_data_continuous_a200.pth'
  replay_buffer.load_state_dict(torch.load(replay_buffer_path))
  print(f'Replay buffer loaded from {replay_buffer_path}')
  print('sample state:', replay_buffer.state[0:5])
  print('sample action:', replay_buffer.action[0:5])
  print('sample next state:', replay_buffer.next_state[0:5])
  print('sample task:', replay_buffer.task[0:5])
  print('sample next task:', replay_buffer.next_task[0:5])
  print('sample reward:', replay_buffer.reward[0:5])
  print('sample done:', replay_buffer.done[0:5])
  # assert np.isclose(replay_buffer.state[0:5]+replay_buffer.action[0:5], replay_buffer.next_state[0:5]).all()
  return replay_buffer, state_dim, action_dim, n_task

def load_rat7m(category, device='cuda:0'):
  state_dim = 54
  action_dim = 54
  n_task = 60
  replay_buffer = buffer.ReplayBuffer(state_dim, action_dim, 1000000, device)
  replay_buffer_path = f'./data/replay_buffer_18body_normalized_new_{category}.pth'
  replay_buffer.load_state_dict(torch.load(replay_buffer_path))
  print(f'Replay buffer loaded from {replay_buffer_path}')
  return replay_buffer, state_dim, action_dim, n_task

def load_halfcheetah():
  state_dim = 17
  action_dim = 6
  replay_buffer = buffer.ReplayBuffer(state_dim, action_dim, 1000000)
  replay_buffer_path = f'../../expert_data/halfcheetah/replay_buffer_notask.pth'
  replay_buffer.load_state_dict(torch.load(replay_buffer_path))
  print(f'Replay buffer loaded from {replay_buffer_path}')
  return replay_buffer, state_dim, action_dim

def to_numpy(*args):
  return [x.cpu().detach().numpy() for x in args]

EPS_GREEDY = 0.01

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default=0, type=str)                     
  parser.add_argument("--alg", default="diffsrsac")                     # Alg name (sac, vlsac, spedersac, ctrlsac, mulvdrq, diffsrsac, spedersac)
  parser.add_argument("--env", default="HalfCheetah-v4")          # Environment name
  parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
  parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
  parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
  parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
  parser.add_argument("--hidden_dim", default=256, type=int)      # Network hidden dims
  parser.add_argument("--feature_dim", default=256, type=int)      # Latent feature dim
  parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
  parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
  parser.add_argument("--learn_bonus", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--extra_feature_steps", default=3, type=int)
  parser.add_argument("--lasso_coef", default=1e-3, type=float)
  parser.add_argument("--feature_lr", default=5e-4, type=float)
  parser.add_argument("--policy_lr", default=3e-4, type=float)
  parser.add_argument("--start_timesteps", default=1e3, type=int)
  parser.add_argument("--actor_type", default='gaussian', type=str)      # Actor type
  args = parser.parse_args()

  if args.alg == 'mulvdrq':
    import sys
    sys.path.append('agent/mulvdrq/')
    from agent.mulvdrq.train_metaworld import Workspace, cfg
    cfg.task_name = args.env
    cfg.seed = args.seed
    workspace = Workspace(cfg)
    workspace.train()

    sys.exit()

  # env = gym.make(args.env)
  # eval_env = gym.make(args.env)
  # env.seed(args.seed)
  # eval_env.seed(args.seed)
  # max_length = env._max_episode_steps

  # setup log 
  log_path = f'log/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  summary_writer = SummaryWriter(log_path)
  expert_buffer, state_dim, action_dim, n_task = load_keymoseq('train', args.dir)
  policy_buffer = buffer.ReplayBuffer(state_dim, action_dim, 100000)
  save_path = f'model/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # 
  # state_dim = env.observation_space.shape[0]
  # action_dim = env.action_space.shape[0] 
  # max_action = float(env.action_space.high[0])

  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    # "action_space": gym.spaces.Box(-1, 1, (action_dim,), dtype=np.float32),
    "discount": args.discount,
    # "tau": args.tau,
    # "hidden_dim": args.hidden_dim,
    "directory": args.dir,
    'actor_type': args.actor_type,
  }

  # Initialize policy
  if args.alg == "sac":
    agent = sac_agent.SACAgent(**kwargs)
  elif args.alg == 'vlsac':
    kwargs['extra_feature_steps'] = args.extra_feature_steps
    kwargs['feature_dim'] = args.feature_dim
    agent = vlsac_agent.VLSACAgent(**kwargs)
  elif args.alg == 'ctrlsac':
    kwargs['extra_feature_steps'] = args.extra_feature_steps
    # hardcoded for now
    kwargs['feature_dim'] = 2048  
    kwargs['hidden_dim'] = 1024
    agent = ctrlsac_agent.CTRLSACAgent(**kwargs)
  elif args.alg == 'diffsrsac':
    agent = diffsrsac_agent.DIFFSRSACAgent(**kwargs)
  elif args.alg == 'spedersac':
    kwargs['extra_feature_steps'] = 2
    kwargs['phi_and_mu_lr'] = args.feature_lr
    kwargs['phi_hidden_dim'] = 512
    kwargs['phi_hidden_depth'] = 1
    kwargs['mu_hidden_dim'] = 512
    kwargs['mu_hidden_depth'] = 1
    kwargs['critic_and_actor_lr'] = args.policy_lr
    kwargs['critic_and_actor_hidden_dim'] = 256
    kwargs['feature_dim'] = args.feature_dim
    # kwargs['state_task_dataset'] = replay_buffer.state
    kwargs['lasso_coef'] = args.lasso_coef
    kwargs['n_task'] = n_task
    kwargs['learnable_temperature'] = False
    kwargs['tau'] = args.tau
    kwargs['hidden_dim'] = args.hidden_dim
    agent = spedersac_agent.SPEDERSACAgent(**kwargs)
    # agent = spedersac_agent.QR_IRLAgent(**kwargs)
    # agent = spedersac_agent.SimpleWorldModel(**kwargs)
  elif args.alg == 'value_dice':
    kwargs['critic_and_actor_hidden_dim'] = 256
    kwargs['target_update_period'] = 2
    kwargs['alpha'] = 1
    kwargs['device'] = 'cuda:0'
    agent = spedersac_agent.ValueDICEAgent(**kwargs)
  args_kwargs = {'args': vars(args), 'kwargs': kwargs}
  np.save(os.path.join(save_path, 'args_kwargs.npy'), args_kwargs)
  print(f'Args saved to {os.path.join(save_path, "args_kwargs.npy")}')
  # replay_buffer = buffer.ReplayBuffer(state_dim, action_dim)
  # agent.load_state_dict(torch.load(f'./model/{args.env}/{args.alg}/{args.dir}/{args.seed}/checkpoint_300000.pth'))
  # print(f'Agent loaded from ./model/{args.env}/{args.alg}/{args.dir}/{args.seed}/checkpoint_300000.pth')
  if args.dir.endswith('_fixf') or args.dir.endswith('_finetunef'):
    # pretrained_dir_name = args.dir.replace('_fixf', '').replace('_finetunef', '')
    pretrained_dir_name = 'S_f128_lasso_001_dataset200_discrete'
    pretrained_model_path = f'./model/{args.env}/{args.alg}/{pretrained_dir_name}/{args.seed}/checkpoint_{args.max_timesteps}.pth'
    agent.load_phi_mu(torch.load(pretrained_model_path))
    print(f'Phi Mu loaded from {pretrained_model_path}')
    # actor_dir_name = pretrained_dir_name + '_actor'
    # actor_model_path = f'./model/{args.env}/{args.alg}/{actor_dir_name}/{args.seed}/checkpoint_{args.max_timesteps}.pth'
    agent.load_actor(torch.load(pretrained_model_path))
    print(f'Actor loaded from {pretrained_model_path}')
    # if args.dir.endswith('fixf'):
    #   print('Fix Phi and Mu')
    # else:
    #   print('Finetune Phi')

    
  # if 'actorclone' in args.dir:
  #   pretrained_dir_name = args.dir.replace('_actorclone', '')
  #   pretrained_model_path = f'./model/{args.env}/{args.alg}/{pretrained_dir_name}/{args.seed}/checkpoint_{args.max_timesteps}.pth'
  #   agent.load_phi_mu(torch.load(pretrained_model_path))
  #   print(f'Phi Mu loaded from {pretrained_model_path}')
  #   agent.load_actor(torch.load(pretrained_model_path))
  #   print(f'Actor loaded from {pretrained_model_path}')


  # Evaluate untrained policy
  # evaluations = [util.eval_policy(agent, eval_env)]
  # state, done = env.reset(), False
  # episode_reward = 0
  episode_timesteps = 0
  # episode_num = 0
  timer = util.Timer()
  print('Start training...')
  for t in range(int(args.max_timesteps)):
    
    episode_timesteps += 1
    info = agent.train(expert_buffer, batch_size=args.batch_size)
    # expert_batch = expert_buffer.sample(args.batch_size)
    # state, action, next_state, reward, done, task, next_task = unpack_batch(expert_batch)
    # policy_action = agent.actor.select_action(state)
    # generate_next_state = agent.generate_next_state(state, policy_action)
    # state, policy_action, generate_next_state, reward, done, task, next_task = to_numpy(state, policy_action, generate_next_state, reward, done, task, next_task)
    # policy_buffer.add_batch(state, policy_action, generate_next_state, reward, done, task, next_task)
    # if t > args.start_timesteps:
    #   info = agent.train(expert_buffer, policy_buffer, batch_size=args.batch_size)
    if (t + 1) % args.eval_freq == 0:
      steps_per_sec = timer.steps_per_sec(t+1)


      for key, value in info.items():
        summary_writer.add_scalar(f'info/{key}', value, t+1)
      summary_writer.flush()

      print('Step {}. Steps per sec: {:.4g}.'.format(t+1, steps_per_sec))
      if (t + 1) % (50 * args.eval_freq) == 0:
        print('Saving model...')
        torch.save(agent.state_dict(), f'{save_path}/checkpoint_{t+1}.pth')
        print(f'Model saved at {save_path}/checkpoint_{t+1}.pth')
        # file_pattern = os.path.join(save_path, "checkpoint_*")
        # for file_path in glob.glob(file_pattern):
        #     os.remove(file_path)
        

  summary_writer.close()

  print('Total time cost {:.4g}s.'.format(timer.time_cost()))
