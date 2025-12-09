import retro
import gymnasium as gym
import numpy as np

from env_class import MKII_Single_Env, MKII_obs_space
from single_play_agent import Kombatant


from pathlib import Path


model_check_path = Path('/kombat_artifacts/checkpoints/learner_group/learner/rl_module/default_policy')

conv_layer_spec = [
(32, 8, 4, 0),
(64, 4, 2, 0),
(64, 3, 1, 0),
(64, 1, 1, 0)
]

trained = Kombatant.from_checkpoint(model_check_path)

print('loaded model??')
env_config={
                     'state': 'Level1.JaxVsBaraka',
                     'record_dir': False, #'/kombat_artifacts/recordings',
                     'n_skip_steps': 10,
                     'skip_repeat': True,
                     'reset_delay': 174
                 }

test_env = MKII_Single_Env(config=env_config)
print('created env??')
obs, info = test_env.reset()

logits = trained(obs['image'])
print(logits)
print(logits.shape)




