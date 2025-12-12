# define the environmentclass for single_player MKII
import retro
import gymnasium as gym
from ray.rllib.core.columns import Columns

import numpy as np
from itertools import product
#from pprint import pprint

MKII_obs_space = gym.spaces.Dict({
        'image': gym.spaces.Box(low=0, high=1, shape=(224, 320, 3), dtype=np.float32),
        #'health': gym.spaces.Box(low=0, high=120, shape=(1,), dtype=np.float),
        #'enemy_health': gym.spaces.Box(low=0, high=120, shape=(1,), dtype=np.float),
        #'player_location': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float),
        #'enemy_location': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float),  
        'additional_data': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    })


## wrapper to treat the RetroEnv args as a config dict
class MKII_Single_Env(gym.Env):
    def __init__(self, config=None):

        # string for the starting state
        self.initial_state = config['state']

        # false or a directory path
        self.record_dir = config['record_dir']

        # number of additional steps to take after each action
        self.n_skip_steps = config['n_skip_steps']
        
        # if true, repeat the same action for n_skip_steps
        # otherwise, take a no-op action
        #self.skip_repeat = config['skip_repeat']

        self.reset_delay = config['reset_delay']
        self.health_weights = config['health_weights']

        self.prev_healths = {'p1': None, 'p2': None}

        #self.button_mask = [0, 1, 3, 4, 5, 6, 7, 8]
        # Set up discrete action space
        button_mapping = {'start':3,
                  'up':4,
                  'down':5,
                  'left':6,
                  'right':7,
                  'A':1,
                  'B':0,
                  'C':8}
        multi_inputs = product(['A','B','C'],['up','down','left','right'])
        moves = [[button_mapping[b1], button_mapping[b2]] for b1,b2 in multi_inputs]
        moveset = [[]] + [[item] for item in button_mapping.values()] + moves
        action_dict = dict()

        for i, m in enumerate(moveset):
            action_dict[i] = m

        self.action_dict = action_dict

        # actually create the environment
        self.inner_env = retro.make(game='MortalKombatII-Genesis',
                              render_mode='rgb_array',
                              state = self.initial_state,
                              record = self.record_dir)
        
        self.observation_space = MKII_obs_space
        self.action_space = gym.spaces.Discrete(21)

    def convert_obs(self, obs, info):
        # obs is the tuple output by env.step
        new_obs = {
            # convert uint8 image to float
            'image': obs.astype(np.float32) / 255.0 , 
            'additional_data': np.array([info['health'], info['enemy_health'], info['x_position'], info['y_position'], 
                                         info['enemy_x_position'], info['enemy_y_position']], dtype=np.float32)
        }
        return new_obs

    def reset(self, seed=None, options=None):
        self.inner_env.reset(seed, options)
        # take a numer of no action step to get the first observation where we can give input
        # determined though testing for each state
        action = np.zeros(self.action_space.n)
        for i in range(self.reset_delay):
            obs, reward, terminated, truncated, info = self.inner_env.step(action)

        new_obs = self.convert_obs(obs)
        self.prev_healths = {'p1': info['health'], 'p2': info['enemy_health']}
        return new_obs, info

    
    def convert_dicrete_action(self, discrete_action):
        button_idxs = self.action_dict[discrete_action]
        new_action = np.zeros(12)
        for bi in button_idxs:
            new_action[bi] = 1
        return new_action

    def step(self, action):

        # insert the discrete action into the binary action space
        full_action = self.convert_dicrete_action(action)
        
        

        if self.skip_repeat:
            skip_action = full_action
        else:
            skip_action = np.zeros(self.inner_env.action_space.n)

        num_steps = self.n_skip_steps + 1

        for _ in range(num_steps):
            obs, not_reward, terminated, truncated, info = self.inner_env.step(full_action)
            if terminated or truncated:
                break
                #reward += obs[1]

        new_obs = self.convert_obs(obs, info)

        # calculate reward
        reward = self.health_weights[0] * (info['health'] - self.prev_healths['p1']) + self.health_weights[1] * (info['enemy_health'] - self.prev_healths['p2'])
        # reward winning and penalizing losing
        if info['enemy_health'] <= 0:
            reward = info['health']**2 + 10
        elif info['health'] <= 0:
            reward = -info['enemy_health']**2 - 10

        self.prev_healths = {'p1': info['health'], 'p2': info['enemy_health']}

        return new_obs, reward, terminated, truncated, info
    
    def close(self):
        self.inner_env.close()

    def stop_record(self):
        self.inner_env.stop_record()

    def render(self):
        return self.inner_env.render()
    

if __name__ == "__main__":
    text ='''
    example usage
    
    from ray.rllib.algorithms.ppo import PPOConfig

    config = (
        PPOConfig()
        .environment(env=MKII_Single_Env,
                     env_config={
                         'state': 'Level1.JaxVsBaraka',
                         'record_dir': False
                     }))
    '''
    print(text)