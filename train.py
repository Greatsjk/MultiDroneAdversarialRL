#!/usr/bin/env python
# coding=utf-8

import sys,os
import datetime
import gym
import torch

from ToyLayer import TopLayerEnv,OUNoise
from agent import DDPG
from common.utils import save_results,make_dir
from common.plot import plot_rewards


class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG'
        self.result_path = './outputs/results/'  # path to save results
        self.model_path = './outputs/models/'  # path to save results
        self.gamma = 0.99
        self.critic_lr = 1e-3
        self.actor_lr = 1e-4
        self.memory_capacity = 10000
        self.batch_size = 128
        self.train_eps = 30000
        self.eval_eps = 50
        self.eval_steps = 200
        self.target_update = 4
        self.hidden_dim = 30
        self.soft_tau = 1e-2
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

def env_agent_config(cfg,seed=1):
    env = TopLayerEnv()
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim,action_dim,cfg)
    return env,agent

def train(cfg, env, agent):
    print('Start to train ! ')
    print(f'Algorithm:{cfg.algo}, Device:{cfg.device}')
    ou_noise = OUNoise(env.action_space)  # action noise
    rewards = []
    ma_rewards = []  # moving average rewards
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            env.render()
            i_step += 1
            action = agent.choose_action(state)
            
            action = ou_noise.get_action(
                action, i_step)  # 即paper中的random process
            next_state, reward, done, _ = env.step(action)
            # print(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        print('Episode:{}/{}, Reward:{}'.format(i_episode+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training!')
    return rewards, ma_rewards

def eval(cfg, env, agent):
    print('Start to Eval ! ')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    for i_episode in range(cfg.eval_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            env.render()
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
        print('Episode:{}/{}, Reward:{}'.format(i_episode+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete Eval!')
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = DDPGConfig()

    # train
    env,agent = env_agent_config(cfg,seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="train",
                 algo=cfg.algo, path=cfg.result_path)
    
    # eval
    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=cfg.model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='eval',path=cfg.result_path)
    plot_rewards(rewards,ma_rewards,tag="eval",env=cfg.env,algo = cfg.algo,path=cfg.result_path)