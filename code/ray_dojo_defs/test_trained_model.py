import retro
import gymnasium as gym
import numpy as np

from env_class import MKII_Single_Env, MKII_obs_space
from single_play_agent import Kombatant


from pathlib import Path


model_check_path = Path('/scratch/mcg4aw/kombat_artifacts/checkpoints/learner_group/learner/rl_module/default_policy')


trained = Kombatant.from_checkpoint(model_check_path)


env_config={
                     'state': 'Level1.JaxVsBaraka',
                     'record_dir': False,
                     'n_skip_steps': 10,
                     'skip_repeat': True,
                     'reset_delay': 100
                 }

test_env = MKII_Single_Env(config=env_config)

obs, info = test_env.reset()

