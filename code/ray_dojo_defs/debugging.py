
import gymnasium as gym
import retro
import numpy as np

import gymnasium.utils.passive_env_checker as pec

from env_class import MKII_Single_Env, MKII_obs_space

env_config={
                     'state': 'Level1.JaxVsBaraka',
                     'record_dir': False,
                     'n_skip_steps': 10,
                     'skip_repeat': True,
                     'reset_delay': 100
                 }


test_env = MKII_Single_Env(config=env_config)

obs, info = test_env.reset()


assert isinstance(MKII_obs_space, gym.spaces.Dict), "MKII_obs_space is not a Dict"
assert isinstance(MKII_obs_space['image'], gym.spaces.Box), 'Image is not a box'
assert isinstance(MKII_obs_space['additional_data'], gym.spaces.Box), 'Additional_data is not a box'

pec.check_obs(obs['image'], MKII_obs_space['image'], 'reset')
pec.check_obs(obs['additional_data'], MKII_obs_space['additional_data'], 'reset')

for key in obs.keys():
    data = obs[key]
    print(key)
    print(f'Data type: {type(data)}')
    print(data)


print('Image shapes:')
print(obs['image'].shape)
print(MKII_obs_space['image'].shape)

print('Additional data shapes:')
print(obs['additional_data'].shape)
print(MKII_obs_space['additional_data'].shape)
