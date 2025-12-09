import retro
import gymnasium as gym
import numpy as np

from env_class import MKII_Single_Env, MKII_obs_space
from single_play_agent import Kombatant


from pathlib import Path
import torch

from ray.rllib.core.distribution.torch.torch_distribution import TorchCategorical
from ray.rllib.core.columns import Columns


model_check_path = Path('/kombat_artifacts/checkpoints/learner_group/learner/rl_module/default_policy')

conv_layer_spec = [
(32, 8, 4, 0),
(64, 4, 2, 0),
(64, 3, 1, 0),
(64, 1, 1, 0)
]

trained = Kombatant.from_checkpoint(model_check_path)

#print('loaded model??')
env_config={
                     'state': 'Level1.JaxVsBaraka',
                     'record_dir': '/kombat_artifacts/recordings',
                     'n_skip_steps': 10,
                     'skip_repeat': True,
                     'reset_delay': 174
                 }

test_env = MKII_Single_Env(config=env_config)
print('created env??')
obs, info = test_env.reset()

img = torch.tensor(obs['image']).unsqueeze(0)
print(img.shape)

additional = torch.tensor(obs['additional_data']).unsqueeze(0)
print(additional.shape)

data = {'obs':{'image': img, 'additional_data': additional}}

output = trained(data)


logits = output[Columns.ACTION_DIST_INPUTS]

dist = TorchCategorical(logits=logits)

action = dist.sample().item()


truncated = False
terminated = False

reward_total = 0

while not terminated and not truncated:
    obs, reward, truncated, terminated, info = test_env.step(action)
    
    # convert obs to tensor
    img = torch.tensor(obs['image']).unsqueeze(0)
    additional = torch.tensor(obs['additional_data']).unsqueeze(0)
    data = {'obs':{'image': img, 'additional_data': additional}}
    # feed to network
    output = trained(data)
    
    # Feed logits to distribution
    dist = TorchCategorical(logits=output[Columns.ACTION_DIST_INPUTS])
    # get action
    action = dist.sample().item()
    # add reward
    reward_total += reward


print(reward_total)
test_env.stop_record()
test_env.close()




