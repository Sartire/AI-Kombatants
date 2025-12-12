#from ray.tune.registry import register_env

from env_class import MKII_Single_Env, MKII_obs_space
from single_play_agent import Kombatant

import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import tune
import ray

import gymnasium as gym
from pprint import pprint
from pathlib import Path
import yaml

from time import time
import pickle


# ------------------------
# SETUP
# ------------------------
# out channels, kernel_size, stride, padding


num_workers = 30

NUM_EPOCHS = 10


ray.init(num_cpus=num_workers, num_gpus=1)

base_storage_path = Path('/kombat_artifacts/checkpoints')

# ------------------------
# ALGORITHM CONFIG
# ------------------------

current_dir = Path(__file__).parent

# load configs
with open(current_dir / 'model_configs.yaml', 'r') as f:
    model_configs = yaml.full_load(f)

with open(current_dir / 'algo_configs.yaml', 'r') as f:
    algo_configs = yaml.full_load(f)

def create_config_from_spec(spec_name):

    spec = algo_configs[spec_name]
    # get the model config

    this_model_config = model_configs[spec['which_model_config']]

    #storage_dir = base_storage_path / spec_name

    config = (
        PPOConfig()
        .environment(env=MKII_Single_Env,
                     env_config={
                         'state': 'Level1.JaxVsBaraka',
                         'record_dir': False,
                         'n_skip_steps': spec['n_skip_steps'],
                         'skip_repeat': True,
                         'reset_delay': 174
                     })
        .rl_module(rl_module_spec=RLModuleSpec(
            module_class=Kombatant,
            observation_space=MKII_obs_space,
            action_space=gym.spaces.Discrete(21),
            model_config= this_model_config
        ))
        .env_runners(num_env_runners = num_workers - 5,
                     # only one emulator can be running per process
                     num_envs_per_env_runner = 1,
                     num_cpus_per_env_runner = 1,
                     gym_env_vectorize_mode = gym.VectorizeMode.ASYNC)
        .learners(num_learners = 1,
                  num_gpus_per_learner = 1)
        .training(
            train_batch_size_per_learner=tune.choice([2000,4000]),
            mini_batch_size=128,

            lr=tune.grid_search([1e-3,1e-4]),
            num_epochs=NUM_EPOCHS,
            use_critic=True,
            use_gae=True,
            lambda_ = tune.grid_search([0.9,0.99]),
            gamma = tune.grid_search([0.9,0.995])
            )
        )
    return config

checkpoint_tracker = dict()

for spec_name in algo_configs.keys():
    config, storage_dir = create_config_from_spec(spec_name)

    spec_start_time = time()

    results = tune.run(
        "PPO",
        config=config.to_dict(),
        storage_path = base_storage_path,
        name = spec_name,
        stop={'training_iteration': 1},
        metric="episode_reward_mean",
        mode="max"
    )
    
    best_trial = results.get_best_trial(metric="episode_reward_mean",mode="max")
    
    best_checkpoint = results.get_best_checkpoint(
    trial=best_trial,
    metric="episode_reward_mean",
    mode="max"
    )

    checkpoint_tracker[spec_name] = best_checkpoint.path

    spec_end_time = time()

    print(f'{spec_name} took {(spec_end_time - spec_start_time)/1} seconds')

    print(f'{spec_name} best checkpoint: {checkpoint_tracker[spec_name]}')


# save the checkpoint tracker

with open(base_storage_path / 'checkpoint_tracker.pkl', 'wb') as f:
    pickle.dump(checkpoint_tracker, f)

ray.shutdown() 


