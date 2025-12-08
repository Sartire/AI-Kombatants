#from ray.tune.registry import register_env

from env_class import MKII_Single_Env, MKII_obs_space
from single_play_agent import Kombatant

import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import tune
import gymnasium as gym
from pprint import pprint

# ------------------------
# SETUP
# ------------------------
# out channels, kernel_size, stride, padding
conv_layer_spec = [
(32, 8, 4, 0),
(64, 4, 2, 0),
(64, 3, 1, 0),
]

num_workers = 10
num_episodes = 4

checkpoint_dir = '/kombat_artifacts/checkpoints'

# ------------------------
# ALGORITHM CONFIG
# ------------------------

config = (
    PPOConfig()
    .environment(env=MKII_Single_Env,
                 env_config={
                     'state': 'Level1.JaxVsBaraka',
                     'record_dir': False,
                     'n_skip_steps': 10,
                     'skip_repeat': True
                 })
    .rl_module(rl_module_spec=RLModuleSpec(
        module_cls=Kombatant,
        observation_space=MKII_obs_space,
        action_space=gym.spaces.MultiBinary(8),
        model_config={
            'hidden_dim': 256,
            'conv_layers_spec': conv_layer_spec,
            
        }
    )
    .env_runners(num_env_runners = num_workers,
                 num_envs_per_runner = 1)
    .learners(num_learners = 1)
    .training(
        train_batch_size_per_learner=1000,
        lr=0.0001,
        num_epochs=10)
    .evaluation(
        # Run one evaluation round every iteration.
        evaluation_interval=1,

        # Create 2 eval EnvRunners in the extra EnvRunnerGroup.
        evaluation_num_env_runners=2,
        # Run evaluation for exactly 10 episodes. Note that because you have
        # 2 EnvRunners, each one runs through 5 episodes.
        evaluation_duration_unit="episodes",
        evaluation_duration=10)
    )
)

# ---------------
# RUN TRAINING!?!?!
# ---------------

ppo = config.build()

for ep in range(num_episodes):
    results = ppo.train()
    pprint(results)

    # save checkpoint
    ppo.save_checkpoint(checkpoint_dir)
