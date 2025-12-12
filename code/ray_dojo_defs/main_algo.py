#from ray.tune.registry import register_env

from env_class import MKII_Single_Env, MKII_obs_space
from single_play_agent import Kombatant
from callbacks import EpisodeReturn

import numpy as np
import pandas as pd

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models import ModelCatalog
from ray import tune
from ray.tune import Tuner
import ray

import gymnasium as gym
from pprint import pprint
from pathlib import Path
import yaml

from time import time
import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type=int, default=30)

# ------------------------
# SETUP
# ------------------------
# out channels, kernel_size, stride, padding


num_workers = parser.parse_args().num_workers

NUM_EPOCHS = 10

NUM_ITERATIONS = 1

VEC_MODE = gym.VectorizeMode.ASYNC


ray.init(num_cpus=num_workers, num_gpus=1,
         _temp_dir='/kombat_artifacts/ray_tmp')

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
        PPOConfig().api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True)
        .environment(env=MKII_Single_Env,
                     env_config={
                         'state': 'Level1.JaxVsBaraka',
                         'record_dir': False,
                         'n_skip_steps': spec['n_skip_steps'],
                         'skip_repeat': True,
                         'reset_delay': 174,
                         'health_weights': (-1,1)
                     })
        .rl_module(rl_module_spec=RLModuleSpec(
            module_class=Kombatant,
            observation_space=MKII_obs_space,
            action_space=gym.spaces.Discrete(21),
            model_config= this_model_config
        ))
        .env_runners(num_env_runners = max(num_workers - 5, 1),
                     # only one emulator can be running per process
                     num_envs_per_env_runner = 1,
                     num_cpus_per_env_runner = 1,
                     num_gpus_per_env_runner = 0,
                     gym_env_vectorize_mode = VEC_MODE)
        .learners(num_learners = 1,
                  num_gpus_per_learner = 1)
        .training(
            train_batch_size_per_learner=10000,
            minibatch_size=128,

            lr=1e-4,
            num_epochs=NUM_EPOCHS,
            use_critic=True,
            use_gae=True,
            lambda_ = 0.99,
            gamma = 0.995)
        .callbacks(callbacks_class= EpisodeReturn)
            
        )
    return config

checkpoint_tracker = dict()

pprint(algo_configs)
pprint(model_configs)

import os

#os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

for spec_name in algo_configs.keys():
    config = create_config_from_spec(spec_name)

    spec_start_time = time()

   

    tuner = Tuner("PPO",
        param_space=config.to_dict(),
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean",
            mode="max",
            num_samples=2,  # Number of trials
        ),
        run_config=tune.RunConfig(
            stop={"training_iteration": NUM_ITERATIONS},
            storage_path = base_storage_path,
            name = spec_name,
            checkpoint_config=tune.CheckpointConfig(num_to_keep=3,
                                                    checkpoint_score_attribute='env_runners/episode_return_mean',
                                                    checkpoint_score_order='max')
        )
    )
    
    results = tuner.fit()

    result_df = results.get_dataframe()
    result_df.to_csv(base_storage_path / spec_name/ f'{spec_name}_results.csv')
    

    best_result = results.get_best_result(metric="env_runners/episode_return_mean",mode="max")
    
    br_path = best_result.path
    br_cps = best_result.best_checkpoints

    checkpoint_tracker[spec_name] = {'path': br_path, 'checkpoints': [cp.path for cp in br_cps if cp is not None]}

    spec_end_time = time()

    print(f'{spec_name} took {(spec_end_time - spec_start_time)/1} seconds')

    print(f'{spec_name} best checkpoint: {checkpoint_tracker[spec_name]}')


# save the checkpoint tracker

with open(base_storage_path / 'checkpoint_tracker.pkl', 'wb') as f:
    pickle.dump(checkpoint_tracker, f)

ray.shutdown() 


