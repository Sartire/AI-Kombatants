from ray import tune
import os

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

from env_class import MKII_Single_Env, MKII_obs_space
from single_play_agent import Kombatant
from callbacks import EpisodeReturn
import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models import ModelCatalog
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

num_workers = parser.parse_args().num_workers

NUM_EPOCHS = 10

NUM_ITERATIONS = 1

VEC_MODE = gym.VectorizeMode.ASYNC

ray.init(num_cpus=num_workers, num_gpus=1,
         _temp_dir='/kombat_artifacts/ray_tmp')

base_storage_path = Path('/kombat_artifacts/checkpoints')
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
            train_batch_size_per_learner=2000,
            minibatch_size=128,

            lr=1e-3,
            num_epochs=NUM_EPOCHS,
            use_critic=True,
            use_gae=True,
            lambda_ = 0.99,
            gamma = 0.995
            )
        .evaluation(
        evaluation_interval=1,
        evaluation_num_workers = 1,

        evaluation_duration = 10,
        evaluation_duration_unit = 'episode'
        )
        .callbacks(callbacks_class= EpisodeReturn)
       
        
    )
    return config

config = create_config_from_spec('no_skip_norm')

tuner = Tuner("PPO",
    config=config.to_dict(),
    stop={"training_iteration": 2},
    storage_path = base_storage_path,
    name = 'debug')

results = tuner.fit()



# Print all metrics from first trial
trial = results.trials[0]
if trial.last_result:
    print("Available metrics:")
    for key in sorted(trial.last_result.keys()):
        if "reward" in key.lower() or "return" in key.lower() or "episode" in key.lower():
            print(f"  {key}: {trial.last_result[key]}")

pprint(results.keys())

x = trial.last_result
x['config'] = None
pickle.dump(x, open('/kombat_artifacts/debug_metrics.p', 'wb'))

pprint(x)

ray.shutdown()