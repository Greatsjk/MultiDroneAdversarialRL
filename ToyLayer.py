import gym
from gym import spaces
import numpy as np
import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.classic_control import rendering
from absl import app
from absl import flags
from absl import logging
SHOOTDISTACNE = 15
DANGERDISTACNE = 10
Param_Dict = {'forward':0.2,'shoot':0.5,'danger':0.5}
class _Agent(object):
    def __init__(self,StartLocation):
        self.nest_pose = StartLocation
class TopLayerEnv(gym.Env):
    def __init__(self, map_size=(40, 40)):
        self.index_i = 0
        self.index_j = 0
        self.done = False
        self.reach = False
        self.map = -1 * np.ones(map_size)
        self.current = [10, 10]
        self.target = [20, 20]
        self.nest_pose = [10, 10]
        self.viewer = rendering.Viewer(400, 400)
        self.EAgent = ( _Agent([30,35]),_Agent([35,30]),_Agent([33,33]),_Agent([35,35]) )
        self.distance_last  = np.linalg.norm(np.array(self.EAgent[3].nest_pose) - np.array(self.nest_pose))
        self.action_space = spaces.Box(
            low=-1,
            high=1, shape=(3,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-100,
            high=100, shape=(4,),
            dtype=np.float32
        )
        self.shoot = 0
        self.reward = 0
    def reward_shape(self):
        info = {}
        self.reward = 0
        self.distance = np.linalg.norm(np.array(self.EAgent[3].nest_pose) - np.array(self.nest_pose) )
        #print("self.distance",self.distance)
        info["forward"] = Param_Dict["forward"] * (self.distance_last - self.distance)
        info["shoot"] = Param_Dict["shoot"] * (SHOOTDISTACNE - self.distance)
        if self.shoot == 1 and (SHOOTDISTACNE - self.distance) > 0:
            self.reward += 20
            self.done = 1
            print("good shoot")
        elif self.shoot == 1 and (SHOOTDISTACNE - self.distance) < 0:
            print("bad shoot")
            self.reward += -10
        info["danger"] = - Param_Dict["danger"] * (DANGERDISTACNE - self.distance)
        self.distance_last = self.distance
        for key in Param_Dict.keys():
            if key in info.keys():
                # print(key,info[key])
                self.reward += info[key]
        return self.reward
    def step(self, action):
        self.index_i = self.index_i + action[0]
        self.index_j = self.index_j + action[1]
        info = {}
        if np.fabs(self.index_i) > 19 or np.fabs(self.index_j) > 19:
            self.index_i = float(np.clip(self.index_i, -19, 19))
            self.index_j = float(np.clip(self.index_j, -19, 19))
        for i in range(4):
            self.EAgent[i].nest_pose[0]  = self.EAgent[i].nest_pose[0] - 1
            self.EAgent[i].nest_pose[1]  = self.EAgent[i].nest_pose[1] - 1
        
        self.nest_pose = [self.current[0] + self.index_i, self.current[1]+self.index_j]
        self.nest_pose[0] = float(np.clip(self.nest_pose[0], 0 , 40))
        self.nest_pose[1] = float(np.clip(self.nest_pose[1], 0 , 40))  
        

        if action[2] > 0:
            # self.done = True
            self.shoot = 1
        reward = self.reward_shape()
        if DANGERDISTACNE - self.distance > 0:
            self.done = True
            print("so danger")
        obs = self.obs_shape()
        return obs, reward, self.done, info
    def obs_shape(self):
        return np.concatenate(
          (
              self.nest_pose,   # 2
              self.EAgent[3].nest_pose,  # 2
          ),
          axis=None,
      )
    def reset(self):
        self.index_i = 0
        self.index_j = 0
        self.EAgent[3].nest_pose = [33,33]
        self.done = False
        self.reach = False
        return self.obs_shape()
    def render(self, mode="human"):
        for i in range(41):
            self.viewer.draw_line((0, 10*i), (400, 10*i))
        for j in range(41):
            self.viewer.draw_line((10*j, 0), (10*j, 400))

        for i in range(40):
            for j in range(40):
                if self.map[39 - i, j] < -10:
                    self.viewer.draw_circle(5, color=(0, 0, 0)).add_attr(
                        rendering.Transform(translation=(j * 10 + 5, i * 10 + 5)))

        self.viewer.draw_circle(5, color=(0.8, 0.6, 0.4)).add_attr(
            rendering.Transform(translation=(int(self.nest_pose[0]) * 10,int(self.nest_pose[1]) * 10)))
        
        for i in range(4):
            self.viewer.draw_circle(5, color=(0, 0, 0)).add_attr(
                rendering.Transform(translation=(int(self.EAgent[i].nest_pose[0] )* 10,int(self.EAgent[i].nest_pose[1]) * 10)))
        
        return self.viewer.render(return_rgb_array=mode == 'human')


class NormalizedActions(gym.ActionWrapper):
    ''' 将action范围重定在[0.1]之间
    '''
    def action(self, action):
        
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        
        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action


class OUNoise(object):
    '''Ornstein–Uhlenbeck
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu
        
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs
    
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_obs, self.low, self.high)


def main(_):
    env = TopLayerEnv()
    done = False
    obs = env.reset()
    # env.reset(-1*np.ones(map_size),[30, 30])
    while not done:
       action = [np.random.uniform(-2,3),np.random.uniform(-2,3),0]
       obs, reward, done, info = env.step(action) 
       print(obs)
       env.render() 
       time.sleep(1)
       
    print("so danger",done)
if __name__ == "__main__":
    app.run(main)

