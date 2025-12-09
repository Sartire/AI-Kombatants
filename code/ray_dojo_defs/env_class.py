# define the environmentclass for single_player MKII
import retro
import gymnasium as gym
from ray.rllib.core.columns import Columns

import numpy as np
from itertools import product
#from pprint import pprint

'''
MKII_obs_space = gym.spaces.Dict({
    Columns.OBS: gym.spaces.Dict({
        'image': gym.spaces.Box(low=0, high=255, shape=(224, 320, 3), dtype=np.uint8),
        'health': gym.spaces.Box(low=0, high=120, shape=(1,), dtype=np.intp),
        'x_location': gym.spaces.Box(low=0, high=2000, shape=(1,), dtype=np.intp),
        'y_location': gym.spaces.Box(low=0, high=2000, shape=(1,), dtype=np.intp),
        'enemy_health': gym.spaces.Box(low=0, high=120, shape=(1,), dtype=np.intp),
        'enemy_x_location': gym.spaces.Box(low=0, high=2000, shape=(1,), dtype=np.intp),
        'enemy_y_location': gym.spaces.Box(low=0, high=2000, shape=(1,), dtype=np.intp),
        
    }),
    
    Columns.TERMINATEDS : gym.spaces.MultiBinary(1),
    Columns.TRUNCATEDS: gym.spaces.MultiBinary(1),
    Columns.REWARDS: gym.spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)
})
'''
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
        self.skip_repeat = config['skip_repeat']
        self.reset_delay = config['reset_delay']

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
        self.env = retro.make(game='MortalKombatII-Genesis',
                              render_mode='rgb_array',
                              state = self.initial_state,
                              record = self.record_dir)
        
        self.observation_space = MKII_obs_space
        self.action_space = gym.spaces.Discrete(21)

    def convert_obs(self, obs):
        # obs is the tuple output by env.step
        new_obs = {
            # convert uint8 image to float
            'image': obs[0].astype(np.float32) / 255.0 , 
            'additional_data': np.array([obs[4]['health'], obs[4]['enemy_health'], obs[4]['x_position'], obs[4]['y_position'], 
                                         obs[4]['enemy_x_position'], obs[4]['enemy_y_position']], dtype=np.float32),   
            #'health': obs[4]['health'],
            #'enemy_health': obs[4]['enemy_health'],
            #'player_location': np.array([obs[4]['x_position'], obs[4]['y_position']]),
            #'enemy_location': np.array([obs[4]['enemy_x_position'], obs[4]['enemy_y_position']]),
            
            
        }

        reward = obs[1]
        terminated = obs[2]
        truncated = obs[3]
        info = obs[4]
        return new_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env.reset(seed, options)
        # take a no action step to get the first observation
        action = np.zeros(self.action_space.n)
        for i in range(self.reset_delay):
            obs = self.env.step(action)

        new_obs, reward, terminated, truncated, info = self.convert_obs(obs)
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
            skip_action = np.zeros(self.env.action_space.n)

        

        obs = self.env.step(full_action)

        if self.n_skip_steps > 0:
            for _ in range(self.n_skip_steps):
                obs = self.env.step(skip_action)

        output = self.convert_obs(obs)
        return output
    
    def close(self):
        self.env.close()

    def stop_record(self):
        self.env.stop_recording()

    def render(self):
        return self.env.render()
    

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